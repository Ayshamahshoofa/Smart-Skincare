// Splash screen transition
window.onload = () => {
  setTimeout(() => {
    document.getElementById("splash-screen").style.display = "none";
    document.getElementById("main-content").style.display = "block";
    initCamera();
  }, 1500);
};

// Initialize webcam
function initCamera() {
  const video = document.getElementById("video");
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => {
      console.error("Camera access error:", err);
    });
}

// Capture image from video
document.getElementById("capture").addEventListener("click", () => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const capturedImage = document.getElementById("captured-image");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  capturedImage.src = canvas.toDataURL("image/png");
  capturedImage.style.display = "block";
});

// Analyze button logic
document.getElementById("analyze-button").addEventListener("click", async () => {
  const uploadInput = document.getElementById("upload");
  const capturedImage = document.getElementById("captured-image");
  const resultText = document.getElementById("result");
  const recList = document.getElementById("recommendations");
  const backendUrl = "http://127.0.0.1:5000/analyze"; // Flask backend

  let file;

  // Choose between uploaded or captured image
  if (uploadInput.files.length > 0) {
    file = uploadInput.files[0];
  } else if (capturedImage.src && capturedImage.style.display === "block") {
    const res = await fetch(capturedImage.src);
    const blob = await res.blob();
    file = new File([blob], "captured.png", { type: "image/png" });
  } else {
    alert("Please capture or upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    resultText.textContent = "⏳ Analyzing...";
    recList.innerHTML = "";

    const response = await fetch(backendUrl, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      resultText.textContent = `❌ Error: ${data.error}`;
      return;
    }

    resultText.textContent = `✅ Result: ${data.result}`;

    if (data.recommendations && data.recommendations.length > 0) {
      recList.innerHTML = "<h3>🧴 Recommended Products:</h3>";
      data.recommendations.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        recList.appendChild(li);
      });
    } else {
      recList.innerHTML = "<p>No recommendations available.</p>";
    }

  } catch (err) {
    console.error("Error analyzing:", err);
    resultText.textContent = "❌ Failed to analyze image.";
  }
});

// Show selected file name
document.getElementById("upload").addEventListener("change", function () {
  const fileName = this.files.length > 0 ? this.files[0].name : "No file chosen";
  document.getElementById("file-name").textContent = fileName;
});
