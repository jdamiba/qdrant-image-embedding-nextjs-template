const fs = require("fs");
const path = require("path");

const wasmDir = path.join(process.cwd(), "public", "tfjs-backend-wasm");
if (!fs.existsSync(wasmDir)) {
  fs.mkdirSync(wasmDir, { recursive: true });
}

const sourceDir = path.join(
  process.cwd(),
  "node_modules/@tensorflow/tfjs-backend-wasm/dist"
);
const files = fs.readdirSync(sourceDir).filter((f) => f.endsWith(".wasm"));

files.forEach((file) => {
  fs.copyFileSync(path.join(sourceDir, file), path.join(wasmDir, file));
});
