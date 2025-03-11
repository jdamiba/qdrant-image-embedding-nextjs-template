import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import sharp from "sharp";
import { version } from "process";

let model: mobilenet.MobileNet | null = null;

async function loadModel() {
  if (!model) {
    // Use CPU backend instead of WASM
    await tf.setBackend("cpu");
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
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

    const predictions = await mobileNetModel.classify(tfImage);
    console.log("predictions", predictions);

    // Generate embedding
    const embedding = mobileNetModel.infer(tfImage, true);
    const embeddingData = await embedding.data();
    console.log("embeddingData", embeddingData);

    // Cleanup
    tfImage.dispose();
    embedding.dispose();

    return Array.from(embeddingData);
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}
