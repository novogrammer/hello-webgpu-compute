import '../style.scss'
import Timer from '../Timer';
import { initWebGpuAsync } from './utils';



const NUM_ELEMENTS = 1024*1024;
const WORKGROUP = 64;
// Workgroup数 = ceil(NUM_ELEMENTS / workgroup_size)
const DISPATCH_X = Math.ceil(NUM_ELEMENTS / WORKGROUP);

// WGSLカーネル
const shaderCode = /* wgsl */`
struct BufF32 {
  data: array<f32>,
};

@group(0) @binding(0) var<storage, read>       inputBuffer  : BufF32;
@group(0) @binding(1) var<storage, read_write> outputBuffer : BufF32;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i < ${NUM_ELEMENTS}u) {
    outputBuffer.data[i] = inputBuffer.data[i] * 2.0;
  }
}
`;

function makeCommandBuffer(device:GPUDevice,inputBuffer:GPUBuffer,outputBuffer:GPUBuffer,readBuffer:GPUBuffer){

  const module = device.createShaderModule({ code: shaderCode });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' }
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  pass.dispatchWorkgroups(DISPATCH_X);
  pass.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, Math.min(outputBuffer.size,readBuffer.size));
  
  const commandBuffer=encoder.finish();
  return commandBuffer;
}


async function runAsync():Promise<string[]> {
  const messageList:string[]=[];

  const timerInit=new Timer("timerInit");
  const timerCompute=new Timer("timerCompute");
  const timerMap=new Timer("timerMap");

  timerInit.start();
  const device = await initWebGpuAsync();
  timerInit.stop();


  const inputData = new Float32Array(NUM_ELEMENTS);
  for (let i = 0; i < NUM_ELEMENTS; i++) {
    inputData[i] = i + 1;
  }

  const inputBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(inputBuffer.getMappedRange()).set(inputData);
  inputBuffer.unmap();

  const outputBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const readBuffer = device.createBuffer({
    size: inputData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });


  try{
    timerCompute.start();
    const commandBuffer=makeCommandBuffer(device,inputBuffer,outputBuffer,readBuffer);

    device.queue.submit([commandBuffer]);
    await device.queue.onSubmittedWorkDone();
    timerCompute.stop();

    timerMap.start();
    await readBuffer.mapAsync(GPUMapMode.READ);
    timerMap.stop();
    const outputData = new Float32Array(readBuffer.getMappedRange());

    messageList.push(`NUM_ELEMENTS: ${NUM_ELEMENTS}`);
    messageList.push(`inputData[0]: ${inputData[0]}`);
    messageList.push(`outputData[0]: ${outputData[0]}`);

    messageList.push(timerInit.getElapsedMessage());
    messageList.push(timerCompute.getElapsedMessage());
    messageList.push(timerMap.getElapsedMessage());
    readBuffer.unmap();

  }finally{
    inputBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();
    await device.queue.onSubmittedWorkDone();
    device.destroy();
  }
  return messageList;
}

async function mainAsync():Promise<void> {

  const messageElement=document.querySelector<HTMLTextAreaElement>(".p-demo__message");
  if(!messageElement){
    throw new Error("messageElement is null");
  }

  const executeElement=document.querySelector<HTMLButtonElement>(".p-demo__execute");
  if(!executeElement){
    throw new Error("executeElement is null");
  }

  executeElement.addEventListener("click",async ()=>{
    executeElement.disabled = true;
    messageElement.value="computing...";
    try{
      // ウォームアップ + 本計測
      const warmup = await runAsync();
      const main   = await runAsync();
      messageElement.value = ['ウォームアップ', ...warmup, '本計測', ...main].join('\n');
    }catch(error: any){
      alert(error?.message ?? String(error));
      console.error(error);

    }finally{
      executeElement.disabled = false;
    }

  });

}



mainAsync().catch(console.error);


