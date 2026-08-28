# Hai Huang's Academic Homepage

Source for [quixoteh.github.io](https://quixoteh.github.io/), built with Next.js and the [PRISM](https://github.com/xyjoey/PRISM) academic website template.

## Local development

```bash
npm ci
npm run dev
```

Run the complete production build and content-preservation check with:

```bash
npm run check
```

## Content

- `content/config.toml`: site identity, navigation, and feature settings
- `content/bio.md`, `content/about.toml`, `content/news.toml`: homepage sections
- `content/portfolio.toml`: research projects
- `content/publications.toml`, `content/publications.bib`: publications
- `content/teaching.toml`: awards
- `content/cv-json.toml`, `content/cv-json.md`: CV page
- `public/`: profile image and favicon

Pushes to `master` are built, verified, and deployed to GitHub Pages by `.github/workflows/deploy.yml`. The generated `out/` directory is not committed.

PRISM is distributed under the MIT License; see `LICENSE-PRISM` for the template license.
