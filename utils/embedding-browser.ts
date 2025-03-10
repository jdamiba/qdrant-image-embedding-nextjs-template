import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";

let model: mobilenet.MobileNet | null = null;

async function loadModel() {
  if (!model) {
    model = await mobilenet.load();
  }
  return model;
}

export async function generateEmbedding(
  base64Image: string
): Promise<number[]> {
  try {
    // Load model if not already loaded
    const mobileNetModel = await loadModel();

    // Create an HTML image element from base64
    const img = new Image();
    img.src = `data:image/jpeg;base64,${base64Image}`;
    await new Promise((resolve) => (img.onload = resolve));

    // Convert to tensor
    const tfImage = tf.browser.fromPixels(img);

    // Resize image to 224x224 (MobileNet's expected input size)
    const resized = tf.image.resizeBilinear(tfImage, [224, 224]);

    // Generate embedding
    const embedding = mobileNetModel.infer(resized, true);

    // Get the data as array
    const embeddingData = await embedding.data();

    // Cleanup tensors
    tfImage.dispose();
    resized.dispose();
    embedding.dispose();

    return Array.from(embeddingData);
  } catch (error) {
    console.error("Error generating embedding:", error);
    throw error;
  }
}
