export default class Timer {
  private startTime?:number;
  private stopTime?:number;

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
}