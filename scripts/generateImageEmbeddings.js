const { HfInference } = require("@huggingface/inference");
const { QdrantClient } = require("@qdrant/js-client-rest");
const fs = require("fs").promises;
const path = require("path");
const sharp = require("sharp");
const tf = require("@tensorflow/tfjs-node");
const mobilenet = require("@tensorflow-models/mobilenet");

const IMAGES_DIR = path.join(process.cwd(), "public", "images");
const COLLECTION_NAME = "image_vector_embeddings_20250310";

async function preprocessImage(imagePath) {
  // MobileNet expects 3-channel RGB images
  const imageBuffer = await sharp(imagePath)
    .resize(224, 224, { fit: "contain", background: { r: 0, g: 0, b: 0 } })
    .toFormat("jpeg")
    .toBuffer();

  const tfImage = tf.node.decodeImage(imageBuffer, 3);
  return tfImage;
}

async function generateEmbedding(model, image) {
  const activation = model.infer(image, {
    embedding: true,
  });
  const embedding = await activation.data();
  return Array.from(embedding);
}

async function main() {
  try {
    // Initialize Qdrant client with API key
    const qdrant = new QdrantClient({
      url: process.env.QDRANT_URL || "http://localhost:6333",
      apiKey: process.env.QDRANT_API_KEY,
    });

    // Load MobileNet model
    console.log("Loading MobileNet model...");
    const model = await mobilenet.load();
    console.log("MobileNet model loaded");

    // Delete existing collection if it exists
    try {
      await qdrant.deleteCollection(COLLECTION_NAME);
      console.log(`Deleted existing collection ${COLLECTION_NAME}`);
    } catch (error) {
      // Collection might not exist, which is fine
    }

    // Create collection with correct dimensions
    await qdrant.createCollection(COLLECTION_NAME, {
      vectors: {
        size: 1024, // MobileNet embedding size
        distance: "Cosine",
      },
    });
    console.log(`Collection ${COLLECTION_NAME} created`);

    // Get all images from the directory
    const files = await fs.readdir(IMAGES_DIR);
    const imageFiles = files.filter((file) =>
      /\.(jpg|jpeg|png|gif)$/i.test(file)
    );

    // Process each image
    for (const [index, filename] of imageFiles.entries()) {
      const imagePath = path.join(IMAGES_DIR, filename);
      console.log(`Processing ${filename} (${index + 1}/${imageFiles.length})`);

      try {
        // Preprocess and generate embedding
        const image = await preprocessImage(imagePath);
        const embedding = await generateEmbedding(model, image);

        // Upload to Qdrant
        await qdrant.upsert(COLLECTION_NAME, {
          points: [
            {
              id: index + 1,
              vector: embedding,
              payload: {
                filename,
                path: `/images/${filename}`,
              },
            },
          ],
        });

        // Cleanup
        tf.dispose(image);
      } catch (error) {
        console.error(`Error processing ${filename}:`, error);
        continue; // Skip failed images and continue with the next
      }
    }

    console.log("All images processed and uploaded to Qdrant");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main();
