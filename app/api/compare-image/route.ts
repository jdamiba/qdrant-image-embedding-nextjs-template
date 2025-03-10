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

export async function POST(request: NextRequest) {
  try {
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

    // Generate embedding for uploaded image
    const embedding = await generateEmbedding(base64Image);

    // Get similarity score from Qdrant
    const searchResult = await qdrant.search(COLLECTION_NAME, {
      vector: embedding,
      limit: 1,
      with_payload: true,
    });

    const similarity = searchResult[0]?.score || 0;

    return NextResponse.json({
      similarity,
      filename: searchResult[0]?.payload?.filename,
    });
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
