import { NextRequest, NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";
import { generateEmbedding } from "@/utils/embedding-node";

// Define custom error type
type ProcessingError = {
  message: string;
  status?: number;
  cause?: unknown;
};

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

const COLLECTION_NAME = "image_vector_embeddings_20250310";
const VECTOR_SIZE = 1280; // MobileNet embedding size

async function ensureCollection() {
  try {
    // Check if collection exists
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(
      (collection) => collection.name === COLLECTION_NAME
    );

    if (!exists) {
      // Create collection if it doesn't exist
      await qdrant.createCollection(COLLECTION_NAME, {
        vectors: {
          size: VECTOR_SIZE,
          distance: "Cosine",
        },
      });
    }
  } catch (error) {
    console.error("Error ensuring collection exists:", error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  try {
    // Ensure collection exists before proceeding
    await ensureCollection();

    const formData = await request.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      );
    }

    // Convert file to base64
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const base64Image = buffer.toString("base64");

    // Generate embedding
    const embedding = await generateEmbedding(base64Image);

    // Upload to Qdrant
    await qdrant.upsert(COLLECTION_NAME, {
      points: [
        {
          id: Date.now(),
          vector: embedding,
          payload: {
            filename: file.name,
          },
        },
      ],
    });

    return NextResponse.json({ success: true });
  } catch (error: unknown) {
    console.error("Error processing image:", error);

    const processingError: ProcessingError = {
      message:
        error instanceof Error ? error.message : "Error processing image",
      cause: error,
    };

    return NextResponse.json(
      { error: processingError.message },
      { status: 500 }
    );
  }
}
