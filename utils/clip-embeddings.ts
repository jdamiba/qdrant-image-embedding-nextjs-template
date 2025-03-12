import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// Add interface for CLIP output
interface ClipOutput {
  embedding?: number[];
}

export async function getClipEmbedding(imageBase64: string): Promise<number[]> {
  const output = (await replicate.run(
    "krthr/clip-embeddings:1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4",
    {
      input: {
        image: `data:image/jpeg;base64,${imageBase64}`,
        task_type: "embedding",
      },
    }
  )) as ClipOutput;

  // Add debug logging
  console.log("CLIP model output:", {
    type: typeof output,
    isArray: Array.isArray(output),
    structure: output,
    embedding: output.embedding
      ? {
          type: typeof output.embedding,
          isArray: Array.isArray(output.embedding),
          length: Array.isArray(output.embedding)
            ? output.embedding.length
            : "not an array",
        }
      : "no embedding field",
  });

  return output.embedding || (output as unknown as number[]);
}
