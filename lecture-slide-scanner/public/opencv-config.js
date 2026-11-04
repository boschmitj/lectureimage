var Module = {
  onRuntimeInitialized() {
    console.log("OpenCV loaded successfully");
    window.dispatchEvent(new Event("opencv-ready"));
  }
};