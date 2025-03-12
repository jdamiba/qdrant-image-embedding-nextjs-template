import { NextRequest, NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";
import {
  generateEmbedding,
  getModelInfo,
  EmbeddingModelType,
} from "@/utils/embedding-node";
import crypto from "crypto";

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

async function ensureCollection(collectionName: string, vectorSize: number) {
  try {
    // Check if collection exists
    const collections = await qdrant.getCollections();
    const exists = collections.collections.some(
      (collection) => collection.name === collectionName
    );

    if (exists) {
      // Get and log collection details
      const collection = await qdrant.getCollection(collectionName);
      console.log("Existing collection info:", collection);
    } else {
      // Create collection if it doesn't exist
      console.log("Creating new collection:", {
        name: collectionName,
        vectorSize,
      });
      await qdrant.createCollection(collectionName, {
        vectors: {
          size: vectorSize,
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
    // Add logging to debug form data
    console.log("Request headers:", Object.fromEntries(request.headers));

    const formData = await request.formData();
    console.log("Form data entries:", Object.fromEntries(formData.entries()));

    const file = formData.get("image") as File;
    console.log("File info:", {
      name: file?.name,
      size: file?.size,
      type: file?.type,
    });

    const modelType =
      (formData.get("modelType") as EmbeddingModelType) || "MobileNet";

    if (!file) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      );
    }

    // Validate file type and size
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        { error: "Invalid file type. Please upload an image." },
        { status: 400 }
      );
    }

    if (file.size > 10 * 1024 * 1024) {
      // 10MB limit
      return NextResponse.json(
        { error: "File size too large. Maximum size is 10MB." },
        { status: 400 }
      );
    }

    // Get model info
    const modelInfo = getModelInfo(modelType);

    // Ensure collection exists
    await ensureCollection(modelInfo.collectionName, modelInfo.vectorSize);

    // Convert file to base64
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const base64Image = buffer.toString("base64");

    // Generate embedding with the specified model
    const embedding = await generateEmbedding(base64Image, modelType);

    // Add debug logging
    console.log("Embedding info:", {
      length: Array.isArray(embedding) ? embedding.length : "not an array",
      modelType,
      collectionName: modelInfo.collectionName,
      vectorSize: modelInfo.vectorSize,
      type: typeof embedding,
      isArray: Array.isArray(embedding),
      rawEmbedding: embedding, // Log the raw embedding to see its structure
    });

    // Ensure embedding is an array
    const embeddingArray = Array.isArray(embedding)
      ? embedding
      : Object.values(embedding);

    // Validate embedding array before upsert
    if (!embeddingArray || !Array.isArray(embeddingArray)) {
      throw new Error(`Invalid embedding array: ${typeof embeddingArray}`);
    }

    if (embeddingArray.length !== modelInfo.vectorSize) {
      throw new Error(
        `Embedding size mismatch. Expected ${modelInfo.vectorSize}, got ${embeddingArray.length}`
      );
    }

    if (!embeddingArray.every((n) => typeof n === "number" && !isNaN(n))) {
      throw new Error("Embedding contains non-numeric or NaN values");
    }

    // Generate a simple string ID using timestamp and random number
    const pointId = `point_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    const uuid = crypto.randomUUID(); // Keep UUID for reference in payload

    // Create point object for debugging
    const point = {
      id: pointId,
      vector: embeddingArray as number[],
      payload: {
        filename: file.name,
        modelType,
        uuid: uuid,
      },
    };

    console.log("Attempting to upsert point:", {
      collectionName: modelInfo.collectionName,
      pointId,
      vectorLength: embeddingArray.length,
      payloadKeys: Object.keys(point.payload),
    });

    // Upload to Qdrant
    try {
      await qdrant.upsert(modelInfo.collectionName, {
        points: [point],
      });
    } catch (qdrantError: any) {
      console.error("Qdrant upsert error details:", {
        error: qdrantError?.message,
        response: {
          data: JSON.stringify(qdrantError.response?.data),
          status: qdrantError.response?.status,
          statusText: qdrantError.response?.statusText,
        },
        pointDetails: {
          id: point.id,
          vectorLength: point.vector.length,
          vectorSample: point.vector.slice(0, 5),
          vectorType: typeof point.vector[0],
        },
        collection: await qdrant
          .getCollection(modelInfo.collectionName)
          .catch((e) => e.message),
      });
      throw qdrantError;
    }

    // Return both IDs in the response
    return NextResponse.json({
      success: true,
      id: pointId,
      uuid: uuid,
    });
  } catch (error: unknown) {
    console.error("Error details:", {
      name: error instanceof Error ? error.name : "Unknown",
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });

    const processingError: ProcessingError = {
      message:
        error instanceof Error
          ? `Error processing image: ${error.message}`
          : "Error processing image",
      cause: error,
    };

    return NextResponse.json(
      { error: processingError.message },
      { status: 500 }
    );
  }
}
