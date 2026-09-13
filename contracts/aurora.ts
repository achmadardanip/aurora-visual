export type Score = number; // bilangan finite dalam [0,1]; bukan otomatis probabilitas
export type ISODateTime = string; // RFC 3339 dengan timezone; simpan UTC
export type VisualLabel = "Supported" | "Contradicted" | "Unobservable";
export type FactLabel = "Supported" | "Contradicted" | "InsufficientEvidence";
export type Role = "actor" | "action" | "object" | "attribute" | "location"
          | "time" | "quantity" | "relation" | "cause";
export type Mode = "demo" | "live";
export type Warning = { code: string; message: string; component: string };

export interface MediaRef {
  asset_id: string; sha256: string; media_type: string;
  width: number; height: number; uri: string;
}
export interface RunInfo {
  run_id: string; mode: Mode; started_at: ISODateTime;
  finished_at: ISODateTime; status: "completed" | "partial" | "failed";
  versions: Record<string,string>; warnings: Warning[];
}
export interface Atom {
  atom_id: string; statement: string; role: Role;
  subject: string | null; predicate: string; object: string | null;
  qualifiers: {
    negated: boolean; quantity: number | null;
    time: string | null; location: string | null;
  };
  spans: { start: number; end: number }[];
  depends_on: string[]; check_worthiness: Score;
  parser_confidence: Score | null;
}
export interface Region {
  region_id: string; asset_id: string;
  bbox: [number, number, number, number]; // x_min,y_min,x_max,y_max
  score: Score | null; description: string;
}
export interface VisualAssessment {
  atom_id: string; visual_status: VisualLabel;
  probabilities: Record<VisualLabel,number> | null;
  unmatched_mass: Score | null; observability_score: Score | null;
  supporting_regions: Region[]; contradicting_regions: Region[];
  counter_evidence: string | null; rationale: string;
  inference_kind: "trained" | "heuristic" | "fixture" | "unavailable";
}
export interface Analysis {
  run: RunInfo; atom_set_id: string; atomic_claims: Atom[];
  visual_assessments: VisualAssessment[];
  ocr: { text: string; bbox: [number,number,number,number];
         confidence: Score | null; language: string | null }[];
}
export interface Evidence {
  evidence_id: string; atom_ids: string[];
  source: {
    provider: string;
    kind: "fact_check" | "news" | "official" | "web"
        | "image_provenance" | "local_corpus" | "user_supplied";
    url: string | null; title: string; publisher: string | null;
    language: string | null; published_at: ISODateTime | null;
    retrieved_at: ISODateTime;
  };
  content: {
    text: string; excerpt: string; sha256: string;
    status: "full" | "excerpt_only" | "snippet_only" | "unavailable";
  };
  provenance: {
    original_url: string | null; archive_url: string | null;
    first_seen_at: ISODateTime | null; captured_at: ISODateTime | null;
    date_basis: string | null; discovery_method: string;
    duplicate_cluster_id: string; independence_group_id: string;
    temporal_eligible: boolean | null; usage_note: string | null;
    image_matches: { asset_id: string; matched_url: string | null;
      match_type: "exact" | "near_duplicate" | "semantic";
      score: Score | null }[];
  };
  relevance_score: Score | null; credibility_score: Score | null;
  rerank_score: number | null; score_breakdown: Record<string,number>;
}
export interface ForensicSignal {
  signal_id: string;
  target: { kind: "claim_text" | "image" | "ocr_text";
            asset_id: string | null; text_sha256: string | null };
  modality: "image" | "text"; provider: string; model_version: string | null;
  task: "ai_generation_detection";
  status: "ok" | "inconclusive" | "unsupported" | "unavailable" | "failed";
  raw_score: number | null;
  raw_scale: { min: number; max: number; higher_means_ai: boolean } | null;
  ai_generated_score: Score | null; raw_label: string | null;
  assessment: "likely_ai_generated" | "likely_human_or_camera"
            | "uncertain" | "not_assessed";
  calibration_status: "unknown" | "provider_claimed"
                    | "locally_validated" | "not_applicable";
  applicable_language: string | null; limitations: string[];
  analyzed_at: ISODateTime; error_code: string | null;
}
export interface Retrieval {
  run: RunInfo; atom_set_id: string | null;
  evidence_list: Evidence[]; forensic_signals: ForensicSignal[];
  query_log: { query_id: string; atom_ids: string[]; provider: string;
               query: string; duration_ms: number; result_count: number }[];
  provider_status: { provider: string; capability: string;
    status: "ok" | "disabled" | "unconfigured" | "rate_limited"
          | "failed" | "unsupported"; message: string | null }[];
}
export interface EvidenceLink {
  evidence_id: string; atom_id: string;
  stance: "Supports" | "Contradicts" | "NotRelevant" | "Unclear";
  quote: string | null; rationale: string;
}
export interface Calibration {
  status: "calibrated" | "uncalibrated" | "demo_only";
  calibration_id: string | null; method: string | null;
  alpha: number | null; sample_count: number; group: string | null;
  fallback_used: string | null; validity_notes: string[];
}
export interface AtomicDecision {
  atom_id: string; base_label: FactLabel; status: FactLabel;
  probabilities: Record<FactLabel,number> | null;
  confidence_set: FactLabel[] | null;
  abstention_flag: boolean; abstention_reasons: string[];
  evidence_links: EvidenceLink[]; visual_atom_ids: string[];
  explanation: string; calibration: Calibration;
}
export interface Decision {
  run: RunInfo; atom_set_id: string;
  atomic_verdicts: AtomicDecision[];
  base_label: FactLabel; final_verdict: FactLabel;
  probabilities: Record<FactLabel,number> | null;
  confidence_set: FactLabel[] | null;
  abstention_flag: boolean; abstention_reasons: string[];
  calibration: Calibration;
  decision_report: {
    summary: string; key_findings: string[]; unresolved_questions: string[];
    evidence_ids: string[]; forensic_signal_ids: string[];
    limitations: string[]; suggested_next_steps: string[];
    misinformation_category: string | null;
  };
  human_review: {
    reviewer: string; reviewed_at: ISODateTime;
    verdict: FactLabel; reason: string;
  } | null;
}
export interface AuroraBundle {
  schema_version: "1.0.0"; case_id: string; claim_revision: number;
  mode: Mode; created_at: ISODateTime;
  input: {
    claim_text: string; language: string; images: MediaRef[];
    as_of: ISODateTime | null;
  };
  analysis: Analysis | null; retrieval: Retrieval | null;
  decision: Decision | null;
  warnings: Warning[]; extensions: Record<string,unknown>;
}
