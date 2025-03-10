import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import * as mobilenet from "@tensorflow-models/mobilenet";
import path from "path";
import sharp from "sharp";

// Set WASM path to node_modules location with trailing slash
const wasmPath =
  path.join(
    process.cwd(),
    "node_modules",
    "@tensorflow",
    "tfjs-backend-wasm",
    "dist"
  ) + path.sep;

setWasmPaths(wasmPath);

let model: mobilenet.MobileNet | null = null;

async function loadModel() {
  if (!model) {
    await tf.setBackend("wasm");
    model = await mobilenet.load();
  }
  return model;
}

export async function generateEmbedding(
  base64Image: string
): Promise<number[]> {
  try {
    const mobileNetModel = await loadModel();

    // Convert base64 to buffer
    const imageBuffer = Buffer.from(base64Image, "base64");

    // Process image with sharp
    const processedImage = await sharp(imageBuffer)
      .resize(224, 224, { fit: "contain" })
      .raw()
      .toBuffer();

    // Create tensor from raw pixel data
    const tfImage = tf.tensor3d(new Uint8Array(processedImage), [224, 224, 3]);

    // Generate embedding
    const embedding = mobileNetModel.infer(tfImage, true);
    const embeddingData = await embedding.data();

    // Cleanup
    tfImage.dispose();
    embedding.dispose();

    return Array.from(embeddingData);
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}
