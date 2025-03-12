import { NextRequest, NextResponse } from "next/server";
import { QdrantClient } from "@qdrant/js-client-rest";
import {
  generateEmbedding,
  getModelInfo,
  EmbeddingModelType,
} from "@/utils/embedding-node";

// Define custom error type
type ProcessingError = {
  message: string;
  status?: number;
  cause?: unknown;
};

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("image") as File;
    const modelType =
      (formData.get("modelType") as EmbeddingModelType) || "MobileNet";

    if (!file) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      );
    }

    // Get model info
    const modelInfo = getModelInfo(modelType);

    // Convert file to base64
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const base64Image = buffer.toString("base64");

    // Generate embedding
    const embedding = await generateEmbedding(base64Image, modelType);

    // Search in Qdrant
    const qdrant = new QdrantClient({
      url: process.env.QDRANT_URL,
      apiKey: process.env.QDRANT_API_KEY,
    });

    const results = await qdrant.search(modelInfo.collectionName, {
      vector: embedding,
      limit: 1,
    });

    return NextResponse.json({
      similarity: results[0]?.score ?? 0,
    });
  } catch (error: unknown) {
    console.error("Error comparing images:", error);

    const processingError: ProcessingError = {
      message:
        error instanceof Error ? error.message : "Error comparing images",
      cause: error,
    };

    return NextResponse.json(
      { error: processingError.message },
      { status: 500 }
    );
  }
}
