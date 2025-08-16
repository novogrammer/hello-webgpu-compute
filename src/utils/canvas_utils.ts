
export function drawCheckerBoard(canvasElement:HTMLCanvasElement,width:number,height:number){
  canvasElement.width=width;
  canvasElement.height=height;
  const ctx=canvasElement.getContext("2d");
  if(!ctx){
    throw new Error("ctx is null");
  }

  const size = 32;
  for (let y = 0; y < height; y += size) {
    for (let x = 0; x < width; x += size) {
      ctx.fillStyle = ((x / size + y / size) % 2 === 0) ? "black" : "white";
      ctx.fillRect(x, y, size, size);
    }
  }  

}


export function toFloat32Array(uint8ClampedArray:Uint8ClampedArray):Float32Array{
const float32Array=new Float32Array(uint8ClampedArray.length);
  for(let i=0;i<uint8ClampedArray.length;i++){
    const value=uint8ClampedArray[i];
    float32Array[i]=value/255;
  }
  return float32Array;
}

export function toUint8ClampedArray(float32Array:Float32Array):Uint8ClampedArray{
  const uint8ClampedArray=new Uint8ClampedArray(float32Array.length);
  for(let i=0;i<float32Array.length;i++){
    const value=float32Array[i];
  const normalizedValue=Math.min(Math.max(value,0),1);
    uint8ClampedArray[i]=Math.round(normalizedValue*255);
  }
  return uint8ClampedArray;
}

export function getImageData(canvasElement:HTMLCanvasElement):Uint8ClampedArray{
  const ctx=canvasElement.getContext("2d");
  if(!ctx){
    throw new Error("ctx is null");
  }
  const imageData=ctx.getImageData(0,0,canvasElement.width,canvasElement.height);
  return imageData.data;
}

export function showImageData(canvasElement:HTMLCanvasElement,data:Uint8ClampedArray,width:number,height:number){
  canvasElement.width=width;
  canvasElement.height=height;
  const ctx=canvasElement.getContext("2d");
  if(!ctx){
    throw new Error("ctx is null");
  }
  const imageData = new ImageData(data,width,height);
  ctx.putImageData(imageData,0,0);
}