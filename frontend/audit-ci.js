const { execSync } = require('child_process');

// GHSA-vfj7-8cjw-p6xm (CWE-674) is a stack-exhaustion DoS in braces <= 3.0.3 when expanding
// deeply nested glob patterns. http-proxy-middleware depends on micromatch -> braces.
// In SeenemAll, server proxy routes are fixed static strings; no untrusted input is passed to glob matching.
// There is currently no upstream patch available for braces.
const IGNORED_ADVISORIES = new Set(['ghsa-vfj7-8cjw-p6xm']);

let rawOutput;
try {
  rawOutput = execSync('npm audit --json --omit=dev');
} catch (err) {
  rawOutput = err.stdout;
}

if (!rawOutput || !rawOutput.toString().trim()) {
  console.error('ERROR: No valid output received from npm audit. Failing closed.');
  process.exit(1);
}

let report;
try {
  report = JSON.parse(rawOutput.toString());
} catch (e) {
  console.error('ERROR: Failed to parse npm audit JSON output:', e.message);
  process.exit(1);
}

const vulnMap = report.vulnerabilities || {};
const vulnerabilities = Object.values(vulnMap);

if (vulnerabilities.length === 0) {
  console.log('Frontend dependency audit passed: 0 vulnerabilities found.');
  process.exit(0);
}

function getRootAdvisories(vulnName, map, visited = new Set()) {
  if (visited.has(vulnName)) return new Set();
  visited.add(vulnName);

  const vuln = map[vulnName];
  if (!vuln || !Array.isArray(vuln.via)) return new Set();

  const advisories = new Set();
  for (const viaItem of vuln.via) {
    if (typeof viaItem === 'string') {
      const childAdvisories = getRootAdvisories(viaItem, map, visited);
      for (const adv of childAdvisories) {
        advisories.add(adv);
      }
    } else if (viaItem && typeof viaItem === 'object') {
      const targetStr = (viaItem.url || '') + ' ' + (viaItem.title || '');
      const match = targetStr.match(/GHSA-[a-z0-9-]+/i);
      if (match) {
        advisories.add(match[0].toLowerCase());
      }
    }
  }
  return advisories;
}

const unhandled = [];

for (const v of vulnerabilities) {
  const rootAdvisories = getRootAdvisories(v.name, vulnMap);
  // If no advisory could be resolved, or any resolved advisory is not in the ignored list, flag as unhandled
  if (rootAdvisories.size === 0) {
    unhandled.push({ vulnerability: v.name, reason: 'No advisory ID could be resolved' });
    continue;
  }
  const unhandledForThis = Array.from(rootAdvisories).filter((adv) => !IGNORED_ADVISORIES.has(adv));
  if (unhandledForThis.length > 0) {
    unhandled.push({ vulnerability: v.name, unhandledAdvisories: unhandledForThis });
  }
}

if (unhandled.length > 0) {
  console.error('Unhandled security vulnerabilities detected:');
  console.error(JSON.stringify(unhandled, null, 2));
  process.exit(1);
}

console.log(
  'Frontend dependency audit passed (verified all reported vulnerabilities resolve exclusively to unpatched advisory GHSA-vfj7-8cjw-p6xm).'
);
