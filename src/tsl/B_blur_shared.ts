import '../style.scss'

import Timer from '../utils/Timer';
import {
  drawCheckerBoard,
  getImageData,
  showImageData,
  toFloat32Array,
  toUint8ClampedArray,
} from '../utils/canvas_utils';

import { WebGPURenderer, StorageInstancedBufferAttribute } from 'three/webgpu';
import {
  Fn, storage, instanceIndex, // 1D index（WebGL フォールバックで使用）
  int, float, vec4,
  If, Loop, clamp,
  // 共有メモリ＆内蔵ビルトイン（WebGPU 経路）
  workgroupArray, workgroupBarrier,
  workgroupId, localId,
} from 'three/tsl';

// ---- 設定 ----
const ENABLE_FORCE_WEBGL = false;      // true ならフォールバックを強制（ナイーブ）
const SHOW_COMPUTE_SHADER = false;

const WIDTH  = 1024;
const HEIGHT = 1024;
const PIXELS = WIDTH * HEIGHT;

// タイル・半径
const TILE_X = 16;
const TILE_Y = 16;
const RADIUS = 3;                       // 3→ 7x7

// ディスパッチ（WebGPU 経路）
const DISPATCH_X = Math.ceil(WIDTH  / TILE_X);
const DISPATCH_Y = Math.ceil(HEIGHT / TILE_Y);

// 1D 実行（WebGL フォールバック）
// const WG_1D = 64;
// const DISPATCH_1D = Math.ceil(PIXELS / WG_1D);

async function runAsync(cin: HTMLCanvasElement, cout: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const tInit = new Timer('init');
  const tPrep = new Timer('prepare');
  const tComp = new Timer('compute');
  const tRead = new Timer('read');

  // renderer
  tInit.start();
  const renderer = new WebGPURenderer({ forceWebGL: ENABLE_FORCE_WEBGL });
  await renderer.init();
  tInit.stop();

  // 入力準備
  tPrep.start();
  drawCheckerBoard(cin, WIDTH, HEIGHT);
  const inF32 = toFloat32Array(getImageData(cin));            // RGBA F32
  const inputAttr  = new StorageInstancedBufferAttribute(inF32, 4);
  const outputAttr = new StorageInstancedBufferAttribute(new Float32Array(inF32.length), 4);

  // TSL ノード化（PBO 読み取り可、出力は書き込み可）
  const inputNode  = storage(inputAttr,  'vec4', inputAttr.count).setPBO(true).toReadOnly().setName('input');
  const outputNode = storage(outputAttr, 'vec4', outputAttr.count).setName('output');

  const W = int(WIDTH), H = int(HEIGHT), P = int(PIXELS);

  // ---- Kernel 定義 ----
  // WebGPU 経路：タイル＋ハローを共有メモリに協調ロード
  const kernelShared = Fn(() => {
    // 2D 座標（ワークグループ座標とローカル座標）
    const gid  = workgroupId.toVar("gid");         // (wx, wy, wz)
    const lid  = localId.toVar("lid");   // (lx, ly, lz)

    const wx = int(gid.x).toVar("wx");
    const wy = int(gid.y).toVar("wy");
    const lx = int(lid.x).toVar("lx");
    const ly = int(lid.y).toVar("ly");

    // このWGが担当する出力タイルの原点（画像座標）
    const tileOx = wx.mul(int(TILE_X)).toVar("tileOx");
    const tileOy = wy.mul(int(TILE_Y)).toVar("tileOy");

    // PAD = タイル＋ハロー
    const PAD_X = TILE_X + 2 * RADIUS;
    const padX = int(PAD_X).toVar("padX");
    const PAD_Y = TILE_Y + 2 * RADIUS;
    const padY = int(PAD_Y).toVar("padY");

    // 共有メモリ（vec4）: PAD_X * PAD_Y
    const tile = workgroupArray('vec4', PAD_X * PAD_Y);

    // 共有メモリ index 計算
    const tindex = Fn(([lx,ly]:[any,any]) => ly.mul(padX).add(lx));

    // ---- 1) タイル＋ハローを全員でストライド協調ロード ----
    // Y 方向（ly, ly + TILE_Y, ... < PAD_Y）
    // @ts-ignore
    Loop({ start: ly, end: padY, update: int(TILE_Y), condition: '<' }, ({ i }) => {
      const sly = int(i).toVar("sly");
      // X 方向（lx, lx + TILE_X, ... < PAD_X）
      // @ts-ignore
      Loop({ start: lx, end: padX, update: int(TILE_X), condition: '<' }, ({ i }) => {
        const slx = int(i).toVar("slx");

        // 読み元のグローバル座標（境界は clamp）
        const gx = clamp(tileOx.add(slx).sub(int(RADIUS)), int(0), W.sub(1)).toVar("gx");
        const gy = clamp(tileOy.add(sly).sub(int(RADIUS)), int(0), H.sub(1)).toVar("gy");

        const gIndex = gy.mul(W).add(gx).toVar("gIndex"); // ピクセル index
        // VRAM→共有メモリ
        const ti=tindex(slx, sly).toVar("ti");
        tile.element(ti).assign(inputNode.element(gIndex));
      });
    });

    // 全員のロード完了待ち
    workgroupBarrier();

    // ---- 2) 畳み込み（共有メモリから読み） ----
    // 対応する出力画素のグローバル座標
    const gx2 = tileOx.add(lx).toVar("gx2");
    const gy2 = tileOy.add(ly).toVar("gy2");

    // 端の余りスレッドは無効化
    If(gx2.lessThan(W).and(gy2.lessThan(H)), () => {
      const lx0 = lx.add(int(RADIUS)).toVar("lx0");
      const ly0 = ly.add(int(RADIUS)).toVar("ly0");

      const sum = vec4(0).toVar('sum');
      const count = float(0).toVar('count');

      Loop({ start: int(-RADIUS), end: int(RADIUS), condition: '<=' }, ({ i }) => {
        const dy = int(i).toVar("dy");
        Loop({ start: int(-RADIUS), end: int(RADIUS), condition: '<=' }, ({ i }) => {
          const dx = int(i).toVar("dx");
          const ti = tindex(lx0.add(dx), ly0.add(dy)).toVar("ti");
          sum.addAssign(tile.element(ti));
          count.addAssign(1);
        });
      });

      const outIndex = gy2.mul(W).add(gx2).toVar("outIndex"); // ピクセル index
      outputNode.element(outIndex).assign(sum.div(count));
    });
  });

  // WebGL フォールバック：共有メモリなし・ナイーブ（1D）
  const kernelNaive1D = Fn(() => {
    const i = int(instanceIndex).toVar('i');
    If(i.lessThan(P), () => {
      const x = i.mod(W).toVar("x"), y = i.div(W).toVar("y");

      const sum = vec4(0).toVar('sum');
      const count = float(0).toVar('count');

      Loop({ start: int(-RADIUS), end: int(RADIUS), condition: '<=' }, ({ i }) => {
        const dy = int(i).toVar("dy");
        Loop({ start: int(-RADIUS), end: int(RADIUS), condition: '<=' }, ({ i }) => {
          const dx = int(i).toVar("dx");
          const nx = clamp(x.add(dx), 0, W.sub(1)).toVar("nx");
          const ny = clamp(y.add(dy), 0, H.sub(1)).toVar("ny");
          const gIndex = ny.mul(W).add(nx).toVar("gIndex");
          sum.addAssign(inputNode.element(gIndex));
          count.addAssign(1);
        });
      });

      const outputIndex = y.mul(W).add(x).toVar("outputIndex");
      outputNode.element(outputIndex).assign(sum.div(count));
    });
  });

  // 経路分岐：WebGPUなら共有メモリ版、WebGLならナイーブ1D
  const computeNode = ENABLE_FORCE_WEBGL
    ? kernelNaive1D().compute(PIXELS)                            // 1D 実行（インスタンス数=画素数）
    : kernelShared().computeKernel([TILE_X, TILE_Y, 1]); // 2D WG

  tPrep.stop();

  // 実行
  tComp.start();
  if (ENABLE_FORCE_WEBGL) {
    await renderer.computeAsync(computeNode);                    // 1D はサイズ指定不要
  } else {
    await renderer.computeAsync(computeNode, [DISPATCH_X, DISPATCH_Y, 1]);
  }
  tComp.stop();

  if (SHOW_COMPUTE_SHADER) {
    try {
      // 内部APIのため try/catch
      // @ts-ignore
      lines.push((renderer as any)._nodes.getForCompute(computeNode).computeShader);
    } catch {}
  }

  // 読み戻し→表示
  tRead.start();
  const outBuf = await renderer.getArrayBufferAsync(outputAttr);
  const outF32 = new Float32Array(outBuf);
  tRead.stop();

  showImageData(cout, toUint8ClampedArray(outF32), WIDTH, HEIGHT);

  lines.push(`B_blur_shared (TSL) WIDTH×HEIGHT=${WIDTH}×${HEIGHT}, R=${RADIUS}`);
  lines.push(tInit.getElapsedMessage());
  lines.push(tPrep.getElapsedMessage());
  lines.push(tComp.getElapsedMessage());
  lines.push(tRead.getElapsedMessage());

  renderer.dispose();
  return lines;
}

// ---- UI ひな形 ----
async function mainAsync(): Promise<void> {
  const msg  = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  const btn  = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  const cin  = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--input');
  const cout = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--output');
  if (!msg || !btn || !cin || !cout) throw new Error('elements not found');

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    msg.value = 'computing...';
    try {
      const warmup = await runAsync(cin, cout);
      const main   = await runAsync(cin, cout);
      msg.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    } catch (e: any) {
      alert(e?.message ?? String(e));
      console.error(e);
    } finally {
      btn.disabled = false;
    }
  });
}

mainAsync().catch(console.error);
