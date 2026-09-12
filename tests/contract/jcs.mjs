import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import canonicalize from '../../frontend/node_modules/canonicalize/lib/canonicalize.js';
const vector=JSON.parse(fs.readFileSync(new URL('../../contracts/jcs-golden.json',import.meta.url),'utf8'));
const canonical=canonicalize(vector.input);
assert.equal(canonical,vector.canonical_utf8);
assert.equal(crypto.createHash('sha256').update(canonical).digest('hex'),vector.sha256);
const b=vector.input;
const atoms=structuredClone(b.analysis.atomic_claims).sort((a,b)=>a.atom_id.localeCompare(b.atom_id));
for(const a of atoms){a.depends_on.sort();a.spans.sort((a,b)=>a.start-b.start||a.end-b.end);}
const data={case_id:b.case_id,claim_revision:b.claim_revision,claim_text_sha256:crypto.createHash('sha256').update(b.input.claim_text).digest('hex'),image_sha256:b.input.image?.sha256??null,atomic_claims:atoms};
assert.equal('aset_'+crypto.createHash('sha256').update(canonicalize(data)).digest('hex'),b.analysis.atom_set_id);
console.log('Python / JavaScript JCS bytes, hash, and atom_set_id match');
