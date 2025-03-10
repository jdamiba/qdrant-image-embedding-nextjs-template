# Image Similarity Search with TensorFlow.js and Qdrant

This application demonstrates how to implement image similarity search using TensorFlow.js for generating vector embeddings and Qdrant for similarity search.

## Technical Overview

### Vector Embeddings with TensorFlow.js

- Uses MobileNet model through TensorFlow.js to generate vector embeddings (1024-dimensional vectors) from images
- Leverages WebAssembly (WASM) backend for efficient tensor operations
- The embedding process:
  1. Resizes images to 224x224 pixels (MobileNet's expected input size)
  2. Passes the image through MobileNet to generate embeddings
  3. Uses the model's feature vectors as embeddings for similarity comparison

### Similarity Search with Qdrant

- Stores image embeddings in a Qdrant collection with Cosine similarity metric
- Each vector in the collection includes:
  - 1024-dimensional embedding vector
  - Metadata (filename)
- Performs similarity search using Qdrant's vector search capabilities
- Returns similarity scores between 0 and 1 (higher means more similar)

## Getting Started

1. Set up environment variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

2. Install dependencies:

```bash
npm install
```

This will also run a postinstall script to set up WebAssembly files required by TensorFlow.js.

3. Run the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser.

## How It Works

1. **Adding Images to Collection**

   - Upload one or more images
   - TensorFlow.js generates embeddings using MobileNet
   - Embeddings are stored in Qdrant collection

2. **Finding Similar Images**
   - Upload a target image
   - Generate its embedding
   - Query Qdrant to find similarity scores
   - Display the similarity percentage

## Technical Details

### WebAssembly Integration

- TensorFlow.js uses WebAssembly for efficient tensor operations
- WASM files are copied from node_modules to public directory during installation
- This enables high-performance image processing in both browser and server environments

### Vector Search

- Collection uses Cosine similarity for comparing vectors
- Each vector has 1024 dimensions (MobileNet's feature vector size)
- Similarity scores are normalized between 0 and 1

## Learn More

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [MobileNet Model](https://github.com/tensorflow/tfjs-models/tree/master/mobilenet)

##Qdrant Image Embedding Template

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
