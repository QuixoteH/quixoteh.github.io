import fs from 'node:fs';
import path from 'node:path';

const redirects = {
  'about/index.html': '/',
  'about.html': '/',
  'portfolio/2025-msm-seg/index.html': '/portfolio/',
  'portfolio/2026-marso/index.html': '/portfolio/',
  'portfolio/2026-mujoco-playground/index.html': '/portfolio/',
  'portfolio/2026-pest-detection/index.html': '/portfolio/',
  'portfolio/2026-so101/index.html': '/portfolio/',
  'publication/2025-msm-seg/index.html': '/publications/',
  'award/2024-kaggle-home-credit/index.html': '/teaching/',
};

for (const [file, destination] of Object.entries(redirects)) {
  const output = path.join(process.cwd(), 'out', file);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=${destination}">
    <link rel="canonical" href="${destination}">
    <title>Redirecting...</title>
  </head>
  <body>
    <p>This page has moved to <a href="${destination}">${destination}</a>.</p>
  </body>
</html>
`);
}

console.log(`Created ${Object.keys(redirects).length} compatibility redirects.`);
