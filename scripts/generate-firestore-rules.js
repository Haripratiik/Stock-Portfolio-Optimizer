#!/usr/bin/env node
/**
 * Generate firestore.rules from the template.
 * Injects OWNER_EMAIL from env (or .env) so the real email is never committed.
 *
 * Usage:
 *   OWNER_EMAIL=you@example.com node scripts/generate-firestore-rules.js
 *   # or add OWNER_EMAIL to .env and run:
 *   node scripts/generate-firestore-rules.js
 */

const fs = require('fs')
const path = require('path')

// Load .env from project root if it exists
const envPath = path.join(__dirname, '..', '.env')
if (fs.existsSync(envPath)) {
  const env = fs.readFileSync(envPath, 'utf8')
  for (const line of env.split('\n')) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/)
    if (m && !process.env[m[1]]) {
      const val = m[2].replace(/^["']|["']$/g, '').trim()
      process.env[m[1]] = val
    }
  }
}

const ownerEmail = process.env.OWNER_EMAIL
if (!ownerEmail) {
  console.error(
    '[generate-firestore-rules] OWNER_EMAIL is required.\n' +
    '  Set it in .env or run: OWNER_EMAIL=you@example.com node scripts/generate-firestore-rules.js'
  )
  process.exit(1)
}

const templatePath = path.join(__dirname, '..', 'firestore.rules.template')
const outputPath = path.join(__dirname, '..', 'firestore.rules')

const template = fs.readFileSync(templatePath, 'utf8')
const output = template.replace(/\{\{OWNER_EMAIL\}\}/g, ownerEmail)
fs.writeFileSync(outputPath, output, 'utf8')

console.log('[generate-firestore-rules] Wrote firestore.rules (owner: ***@***)')
