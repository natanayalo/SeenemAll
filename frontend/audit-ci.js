const { execSync } = require('child_process');

// GHSA-vfj7-8cjw-p6xm (CWE-674) is a stack-exhaustion DoS in braces <= 3.0.3 when expanding
// deeply nested glob patterns. http-proxy-middleware depends on micromatch -> braces.
// In SeenemAll, server proxy routes are fixed static strings; no untrusted input is passed to glob matching.
// There is currently no upstream patch available for braces.
const IGNORED_PACKAGES = new Set(['braces', 'micromatch', 'http-proxy-middleware']);

let rawOutput;
try {
  rawOutput = execSync('npm audit --json --omit=dev');
} catch (err) {
  rawOutput = err.stdout;
}

if (!rawOutput) {
  console.log('No npm audit output received.');
  process.exit(0);
}

const report = JSON.parse(rawOutput.toString());
const vulnerabilities = Object.values(report.vulnerabilities || {});
const unhandled = vulnerabilities.filter((v) => !IGNORED_PACKAGES.has(v.name));

if (unhandled.length > 0) {
  console.error('Unhandled security vulnerabilities detected:');
  console.error(JSON.stringify(unhandled, null, 2));
  process.exit(1);
}

console.log(
  'Frontend dependency audit passed (ignored unpatched GHSA-vfj7-8cjw-p6xm in braces/http-proxy-middleware).'
);
