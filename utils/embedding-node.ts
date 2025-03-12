import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import sharp from "sharp";
import { getClipEmbedding } from "./clip-embeddings";

// Cache for loaded models
const modelCache: Record<string, any> = {};

export type EmbeddingModelType = "MobileNet" | "CLIP";

export async function loadModel(modelType: EmbeddingModelType) {
  const cacheKey = modelType.toLowerCase();

  if (!modelCache[cacheKey]) {
    // Use CPU backend
    await tf.setBackend("cpu");

    if (modelType === "MobileNet") {
      modelCache[cacheKey] = await mobilenet.load({ version: 2, alpha: 1.0 });
    }
  }

  return modelCache[cacheKey];
}

export async function generateEmbedding(
  base64Image: string,
  modelType: EmbeddingModelType = "MobileNet"
): Promise<number[]> {
  try {
    if (modelType === "CLIP") {
      return await getClipEmbedding(base64Image);
    }

    const model = await loadModel(modelType);

    // Convert base64 to buffer
    const imageBuffer = Buffer.from(base64Image, "base64");

    // Process image with sharp
    const processedImage = await sharp(imageBuffer)
      .resize(224, 224, { fit: "contain" })
      .raw()
      .toBuffer();

    // Create tensor from raw pixel data
    const tfImage = tf.tensor3d(new Uint8Array(processedImage), [224, 224, 3]);

    // Generate embedding with MobileNet
    const embedding = model.infer(tfImage, true);
    const embeddingData = await embedding.data();

    // Cleanup
    tfImage.dispose();
    embedding.dispose();

    return Array.from(embeddingData);
  } catch (error) {
    console.error(`Error generating embedding with ${modelType}:`, error);
    throw error;
  }
}

export function getModelInfo(modelType: EmbeddingModelType) {
  const models = {
    MobileNet: {
      collectionName: "mobilenet_embeddings",
      vectorSize: 1280,
    },
    CLIP: {
      collectionName: "clip_embeddings",
      vectorSize: 768,
    },
  };

  return models[modelType];
}
