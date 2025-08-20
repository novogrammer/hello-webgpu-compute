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

async function runAsync(canvasInput: HTMLCanvasElement, canvasOutput: HTMLCanvasElement): Promise<string[]> {
  const lines: string[] = [];
  const timerInit = new Timer('init');
  const timerPrepare = new Timer('prepare');
  const timerCompute = new Timer('compute');
  const timerRead = new Timer('read');

  // renderer
  timerInit.start();
  const renderer = new WebGPURenderer({
    forceWebGL: ENABLE_FORCE_WEBGL,
  });
  
  await renderer.init();
  const isWebGPUBackend=!!((renderer.backend as any).isWebGPUBackend);
  console.log(`isWebGPUBackend: ${isWebGPUBackend}`);
  timerInit.stop();

  // 入力準備
  timerPrepare.start();
  drawCheckerBoard(canvasInput, WIDTH, HEIGHT);
  const inputData = toFloat32Array(getImageData(canvasInput));            // RGBA F32
  const inputAttribute  = new StorageInstancedBufferAttribute(inputData, 4);
  const outputAttribute = new StorageInstancedBufferAttribute(new Float32Array(inputData.length), 4);

  // TSL ノード化（PBO 読み取り可、出力は書き込み可）
  const inputNode  = storage(inputAttribute,  'vec4', inputAttribute.count).setPBO(true).toReadOnly().setName('input');
  const outputNode = storage(outputAttribute, 'vec4', outputAttribute.count).setName('output');

  const W = int(WIDTH);
  const H = int(HEIGHT);
  const P = int(PIXELS);

  // ---- Kernel 定義 ----
  // WebGPU 経路：タイル＋ハローを共有メモリに協調ロード
  const kernelShared = Fn(() => {
    // 2D 座標（ワークグループ座標とローカル座標）
    const wid  = workgroupId.toVar("gid");         // (wx, wy, wz)
    const lid  = localId.toVar("lid");   // (lx, ly, lz)

    const wx = int(wid.x).toVar("wx");
    const wy = int(wid.y).toVar("wy");
    const lx = int(lid.x).toVar("lx");
    const ly = int(lid.y).toVar("ly");

    // このWGが担当する出力タイルの原点（画像座標）
    const tileOx = wx.mul(int(TILE_X)).toVar("tileOx");
    const tileOy = wy.mul(int(TILE_Y)).toVar("tileOy");

    // PAD = タイル＋ハロー
    const PAD_X = TILE_X + 2 * RADIUS;
    const PAD_Y = TILE_Y + 2 * RADIUS;
    const padX = int(PAD_X).toVar("padX");
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
        const gxRead = clamp(tileOx.add(slx).sub(int(RADIUS)), int(0), W.sub(1)).toVar("gxRead");
        const gyRead = clamp(tileOy.add(sly).sub(int(RADIUS)), int(0), H.sub(1)).toVar("gyRead");

        const gIndex = gyRead.mul(W).add(gxRead).toVar("gIndex"); // ピクセル index
        // VRAM→共有メモリ
        const ti=tindex(slx, sly).toVar("ti");
        tile.element(ti).assign(inputNode.element(gIndex));
      });
    });

    // 全員のロード完了待ち
    workgroupBarrier();

    // ---- 2) 畳み込み（共有メモリから読み） ----
    // 対応する出力画素のグローバル座標
    const gxWrite = tileOx.add(lx).toVar("gxWrite");
    const gyWrite = tileOy.add(ly).toVar("gyWrite");

    // 端の余りスレッドは無効化
    If(gxWrite.lessThan(W).and(gyWrite.lessThan(H)), () => {

      const sum = vec4(0).toVar('sum');
      const count = float(0).toVar('count');

      Loop({ start: int(-RADIUS), end: int(RADIUS), condition: '<=' }, ({ i }) => {
        const dy = int(i).toVar("dy");
        Loop({ start: int(-RADIUS), end: int(RADIUS), condition: '<=' }, ({ i }) => {
          const dx = int(i).toVar("dx");
          const ti = tindex(lx.add(int(RADIUS)).add(dx), ly.add(int(RADIUS)).add(dy)).toVar("ti");
          sum.addAssign(tile.element(ti));
          count.addAssign(1);
        });
      });

      const outIndex = gyWrite.mul(W).add(gxWrite).toVar("outIndex"); // ピクセル index
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
  let computeNode;
  if(isWebGPUBackend){
    // 2D WG
    computeNode=kernelShared().computeKernel([TILE_X, TILE_Y, 1]);
  }else{
    // 1D 実行（インスタンス数=画素数）
    computeNode=kernelNaive1D().compute(PIXELS);
  }

  timerPrepare.stop();

  // 実行
  timerCompute.start();
  if (isWebGPUBackend) {
    await renderer.computeAsync(computeNode, [DISPATCH_X, DISPATCH_Y, 1]);
  } else {
    await renderer.computeAsync(computeNode);
  }
  timerCompute.stop();

  if (SHOW_COMPUTE_SHADER) {
    lines.push((renderer as any)._nodes.getForCompute(computeNode).computeShader);
  }

  // 読み戻し→表示
  timerRead.start();
  const outputBuffer = await renderer.getArrayBufferAsync(outputAttribute);
  const outputData = new Float32Array(outputBuffer);
  timerRead.stop();

  showImageData(canvasOutput, toUint8ClampedArray(outputData), WIDTH, HEIGHT);

  lines.push(`B_blur_shared (TSL) WIDTH×HEIGHT=${WIDTH}×${HEIGHT}, R=${RADIUS}`);
  lines.push(timerInit.getElapsedMessage());
  lines.push(timerPrepare.getElapsedMessage());
  lines.push(timerCompute.getElapsedMessage());
  lines.push(timerRead.getElapsedMessage());

  renderer.dispose();
  return lines;
}

// ---- UI ひな形 ----
async function mainAsync(): Promise<void> {
  const messageElement = document.querySelector<HTMLTextAreaElement>('.p-demo__message');
  if(!messageElement){
    throw new Error("messageElement is null");
  }

  const executeElement = document.querySelector<HTMLButtonElement>('.p-demo__execute');
  if(!executeElement){
    throw new Error("executeElement is null");
  }

  const canvasInputElement = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--input');
  if(!canvasInputElement){
    throw new Error("canvasInputElement is null");
  }
  const canvasOutputElement = document.querySelector<HTMLCanvasElement>('.p-demo__canvas--output');
  if(!canvasOutputElement){
    throw new Error("canvasOutputElement is null");
  }

  executeElement.addEventListener('click', async () => {
    executeElement.disabled = true;
    messageElement.value = 'computing...';
    try {
      // ウォームアップ + 本計測
      const warmup = await runAsync(canvasInputElement, canvasOutputElement);
      const main   = await runAsync(canvasInputElement, canvasOutputElement);
      messageElement.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    } catch (error: any) {
      alert(error?.message ?? String(error));
      console.error(error);
    } finally {
      executeElement.disabled = false;
    }
  });
}

mainAsync().catch(console.error);
