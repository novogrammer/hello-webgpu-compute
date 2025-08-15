import '../style.scss'
import Timer from '../Timer';
import { initWebGPUAsync } from './utils';



const NUM_ELEMENTS = 1024*1024;

// WGSLカーネル
const shaderCode = /* wgsl */`
  @group(0) @binding(0) var<storage, read>  inputBuffer : array<f32>;
  @group(0) @binding(1) var<storage, read_write> outputBuffer : array<f32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let i = global_id.x;
    if (i < ${NUM_ELEMENTS}u) {
      outputBuffer[i] = inputBuffer[i] * 2.0;
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

  // Workgroup数 = ceil(NUM_ELEMENTS / workgroup_size)
  pass.dispatchWorkgroups(Math.ceil(NUM_ELEMENTS / 64));
  pass.end();

  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, Math.min(outputBuffer.size,readBuffer.size));
  
  const commandBuffer=encoder.finish();
  return commandBuffer;
}


async function runAsync():Promise<string[]> {
  const messageList:string[]=[];

  const timerInit=new Timer();
  const timerMakeCommand=new Timer();
  const timerCompute=new Timer();
  const timerMap=new Timer();

  timerInit.start();
  const device = await initWebGPUAsync();
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
    timerMakeCommand.start();
    const commandBuffer=makeCommandBuffer(device,inputBuffer,outputBuffer,readBuffer);
    timerMakeCommand.stop();

    timerCompute.start();
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

    messageList.push(`timerInit: ${timerInit.getElapsed()}[ms]`);
    messageList.push(`timerMakeCommand: ${timerMakeCommand.getElapsed()}[ms]`);
    messageList.push(`timerCompute: ${timerCompute.getElapsed()}[ms]`);
    messageList.push(`timerMap: ${timerMap.getElapsed()}[ms]`);
    readBuffer.unmap();

  }finally{
    inputBuffer.destroy();
    outputBuffer.destroy();
    readBuffer.destroy();
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

  executeElement.addEventListener("click",()=>{
    messageElement.value="computing...";
    runAsync().then((messageList)=>{
      messageElement.value=messageList.join("\n");
    }).catch((error)=>{
      alert(error?.message ?? String(error));
      console.error(error);
    });

  });

}



mainAsync().catch((error)=>{
  console.error(error);
})


