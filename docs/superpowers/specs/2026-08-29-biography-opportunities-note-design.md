# Biography Opportunities Note Design

## Goal

Add the user-approved opportunities statement beneath the existing Biography paragraph, following Zhe Li's blockquote treatment without changing the paragraph itself.

## Content

The visible sentence must be exactly:

> I am currently looking for research collaboration and internship opportunities related to Force-aware Robot Learning, Robot Manipulation, and Vision-Language-Action Models.

## Presentation

- Place the sentence immediately after the existing Biography paragraph in `content/bio.md`.
- Use Markdown blockquote syntax so the existing Biography renderer produces the same left-border treatment as the reference site.
- Keep the sentence as one paragraph and preserve the capitalization of the three research interests.

## Constraints

- Do not modify the existing Biography paragraph.
- Do not change the Interests list, profile, navigation, other sections, styling, or component code.
- Do not deploy wording other than the user-approved sentence.

## Verification

- Assert the exact sentence and blockquote marker in the source.
- Require the exact sentence in the exported homepage.
- Verify the deployed desktop and mobile homepage shows the blockquote beneath Biography with no horizontal overflow or console errors.

