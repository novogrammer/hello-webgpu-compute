export default class Timer {
  private label:string;
  private startTime?:number;
  private stopTime?:number;

  constructor(label:string){
    this.label=label;
  }

  start(): void {
    this.startTime = performance.now();
    this.stopTime = undefined;
  }
  stop(): void {
    if (this.startTime === undefined) {
      throw new Error("Timer has not been started.");
    }    
    this.stopTime = performance.now();
  }
  getElapsed(): number {
    if (this.startTime === undefined) {
      throw new Error("Timer has not been started.");
    }
    if (this.stopTime === undefined) {
      throw new Error("Timer has not been stopped.");
    }
    return this.stopTime - this.startTime;
  }
  getElapsedMessage(): string{
    return `${this.label}: ${this.getElapsed()}[ms]`;
  }
}