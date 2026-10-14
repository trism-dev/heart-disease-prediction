import { readFileSync, copyFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Keep this deployable subdirectory self-contained without training at build/runtime.
const destination = new URL('../models/', import.meta.url);
mkdirSync(destination, { recursive: true });
for (const name of ['pipeline.joblib', 'metadata.json']) {
  // Resolve from scripts/ to the original repository model directory.
  const canonical = new URL('../../models/' + name, import.meta.url);
  const target = new URL(name, destination);
  if (process.argv.includes('--check')) {
    if (!readFileSync(canonical).equals(readFileSync(target))) throw new Error(name + ' differs from the canonical model');
  } else copyFileSync(fileURLToPath(canonical), fileURLToPath(target));
}
