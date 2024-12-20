const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const admin = require("firebase-admin");
const { v4: uuidv4 } = require("uuid");

const app = express();
process.env.TF_CPP_MIN_LOG_LEVEL = '2'; // Minimize TensorFlow logs

const upload = multer({
  limits: { fileSize: 1000000 }, // 1MB limit
  fileFilter(req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("File must be an image"));
    }
    cb(null, true);
  },
});

admin.initializeApp({
  credential: admin.credential.cert(require("./serviceAccountKey.json")),
});
const db = admin.firestore();

let model;

async function retryLoadModel(retries = 3, delay = 5000) {
  const modelUrl = "https://storage.googleapis.com/storage-model-abi/model/model.json";
  for (let i = 0; i < retries; i++) {
    try {
      console.log(`Attempt ${i + 1} to load model from ${modelUrl}`);
      model = await tf.loadGraphModel(modelUrl);
      console.log("GraphModel loaded successfully");
      return;
    } catch (error) {
      console.error(`Attempt ${i + 1} failed:`, error.message);
      if (i < retries - 1) await new Promise((res) => setTimeout(res, delay));
    }
  }
  console.error("Failed to load model after multiple attempts.");
  model = null;
}

retryLoadModel();

app.get("/status", (req, res) => {
  res.status(200).json({
    status: model ? "ready" : "not ready",
    message: model ? "Model loaded and ready for predictions" : "Model not loaded yet",
  });
});

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!model) {
      return res.status(503).json({
        status: "fail",
        message: "Model belum dimuat. Silakan cek status model dengan endpoint /status.",
      });
    }

    const { file } = req;
    if (!file) {
      return res.status(400).json({
        status: "fail",
        message: "No file uploaded. Please provide an image for prediction.",
      });
    }

    // Konversi file gambar ke tensor dan normalisasi gambar agar sesuai dengan model
    const mean = tf.tensor([0.485, 0.456, 0.406]);
    const std = tf.tensor([0.229, 0.224, 0.225]);

    const tensor = tf.node
      .decodeImage(file.buffer, 3)
      .resizeBilinear([224, 224])
      .expandDims()
      .toFloat()
      .div(tf.scalar(255)) // Normalisasi pixel
      .sub(mean)
      .div(std);

    console.log("Input tensor shape:", tensor.shape);

    // Gunakan model untuk prediksi
    const predictions = model.predict(tensor).dataSync();
    console.log("Model predictions:", predictions);

    const threshold = 0.7; // Update threshold sesuai analisis
    const prediction = predictions[0];
    const result = prediction > threshold ? "Cancer" : "Non-cancer";
    console.log("Predicted result:", result);

    const suggestion =
      result === "Cancer" ? "Segera periksa ke dokter!" : "Penyakit kanker tidak terdeteksi.";
    const id = uuidv4();
    const createdAt = new Date().toISOString();

    // Simpan hasil prediksi ke Firestore
    await db.collection("predictions").doc(id).set({
      id,
      result,
      suggestion,
      createdAt,
    });

    // Kirim respons ke klien
    res.status(201).json({
      status: "success",
      message: "Model is predicted successfully",
      data: { id, result, suggestion, createdAt },
    });
  } catch (error) {
    console.error("Error during prediction:", error.message, error.stack);
    res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi.",
    });
  }
});


app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }
  res.status(400).json({
    status: "fail",
    message: "Terjadi kesalahan dalam melakukan prediksi",
  });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
