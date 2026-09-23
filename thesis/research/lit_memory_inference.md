# Literature survey: external/indexed memory, KV-memory compression, and MoE routing predictability

Context: Thinker = latent recurrent core + external indexed (hierarchical) memory holding knowledge/params. Hypothesis under test (V4): the hierarchical index self-organizes by domain, so early recurrent iterations fetch domain knowledge once, giving predictable memory prefetch at inference — contrasted with MoE's per-token, less predictable expert routing. Second thread: a model encodes documents into its own KV memory via a special token; another instance uses these self-generated KV memories (+ distractors) as a long-term KB, trained end-to-end.

## Product-key / sparse memory layers (parametric knowledge stores)

- **Large Memory Layers with Product Keys** (Lample, Sablayrolles, Ranzato, Denoyer, Jégou, NeurIPS 2019, arXiv:1907.05242). What: a huge key-value memory layer addressed via product-key decomposition for fast exact NN lookup, insertable into a Transformer. Finding: a 12-layer model + memory layer beats a 24-layer baseline, 2x faster at inference. Relevance: supports the general idea that a large associative memory can substitute for parametric depth/knowledge — the direct ancestor of Thinker's "external indexed memory holds knowledge" design, but the memory here is a flat product-key structure, not hierarchical/domain-organized, and routing is per-token/per-layer, not amortized across recurrent iterations.

- **Memory Layers at Scale** (Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh — Meta FAIR, arXiv:2412.09764, Dec 2024). What: scaled, parallelizable trainable key-value memory layers (up to 128B memory params) added to dense LMs. Finding: memory-augmented models beat dense models with 2x compute and beat MoE at matched compute/params, especially on factual tasks. Relevance: supports "external memory as a cheap way to hold knowledge separately from compute," directly relevant to Thinker's core/memory split; but again flat lookup, no evidence given about temporal/iteration-wise access locality — doesn't address the prefetch-predictability hypothesis.

## Retrieval / non-parametric memory at inference time

- **kNN-LM: Generalization through Memorization** (Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis, ICLR 2020, arXiv:1911.00172). What: interpolates an LM with a kNN lookup over a datastore of (context-embedding, next-token) pairs. Finding: SOTA WikiText-103 perplexity with zero extra training; especially helps rare/factual patterns; datastore swappable for domain adaptation. Relevance: early evidence that non-parametric external memory specializes by domain when the datastore itself is domain-selected — supports the *possibility* of domain-organized memory, but the organization is imposed externally (which corpus you plug in), not learned/self-organized internally as V4 requires.

- **Memorizing Transformers** (Wu, Rabe, Hutchins, Szegedy, ICLR 2022, arXiv:2203.08913). What: augments a Transformer with a non-differentiable kNN memory of cached (k,v) pairs from recent context, retrieved via approximate NN at inference. Finding: improves LM across code/math/books/webtext as memory size grows to 262k tokens; can use newly-defined functions/theorems at test time. Relevance: closest prior art to "instance stores its own KV pairs as long-term memory," but here it's the *same* model reading its own recent past (single-instance, short-horizon), not one model encoding docs into KV for a *separate* instance's long-term KB — Thinker's cross-instance/end-to-end-trained KV-as-memory setup is a step beyond this.

- **RETRO — Improving Language Models by Retrieving from Trillions of Tokens** (Borgeaud et al., DeepMind, ICML 2022, arXiv:2112.04426). What: retrieves text chunks from a frozen 2T-token datastore via chunked cross-attention (CCA), decoupling memorization from parameter count. Finding: matches GPT-3/Jurassic-1 with 25x fewer parameters. Relevance: reinforces "knowledge can live outside weights," background support for Thinker's premise that memory ≠ parameters; not about self-organization or prefetch predictability (retrieval is per-chunk, frozen, non-hierarchical index via BERT embeddings + approximate NN, not domain-clustered by training).

## Latent/parametric long-term memory that updates online

- **MemoryLLM / M+: Extending MemoryLLM with Scalable Long-Term Memory** (Wang et al., ICML 2025, arXiv:2502.00592). What: MemoryLLM compresses history into a fixed-size per-layer latent memory pool (~1B params); M+ adds a co-trained retriever that offloads dropped memory tokens to CPU-resident long-term storage and retrieves them later. Finding: extends effective retention from <20k to >160k tokens at similar GPU overhead. Relevance: directly relevant precedent for "latent memory + a trained retriever selecting what to fetch," structurally close to Thinker's index+core; the retriever here is a flat two-projector similarity search, not a hierarchical/domain-clustered index, so it doesn't test V4's self-organization claim, but its co-trained retrieval mechanism is a template for how Thinker's index could be trained end-to-end.

- **Titans: Learning to Memorize at Test Time** (Behrouz, Zhong, Mirrokni — Google Research, NeurIPS 2025, arXiv:2501.00663). What: a neural long-term memory module trained online at test time via a "surprise"-gated update rule, combined with short-term attention. Finding: outperforms Transformers at matched context, scales to >2M context. Relevance: supports the general viability of a learned, updatable memory module separate from the attention core (architecturally analogous to Thinker's core/memory split), but memory here is a single associative module updated by surprise, not an indexed/hierarchical structure being probed for domain clustering — orthogonal to V4 but a strong precedent for "core writes to memory at test time."

- **LM2: Large Memory Models** (Kang et al., arXiv:2502.06049, Feb 2025). What: decoder-only Transformer + auxiliary memory module updated via cross-attention and gating, aimed at multi-hop/relational reasoning over long context. Finding: beats RMT by 37% and Llama-3.2 by 86% on BABILong. Relevance: another core+memory dual-pathway architecture supporting the general design; no analysis of whether the memory content organizes by domain or of access locality across steps.

## Compressing context/knowledge into cache-like objects (relevant to "encode docs into KV")

- **Gisting: Learning to Compress Prompts with Gist Tokens** (Mu, Li, Goodman, NeurIPS 2023, arXiv:2304.08467). What: trains a model to compress a prompt into a handful of "gist" tokens (cached KV) via masked attention, reusable across queries. Finding: up to 26x prompt compression, ~40% FLOPs reduction, minimal quality loss. Relevance: a template mechanism for "encode input into compact KV to be reused" — directly analogous to Thinker's "special token → document's KV memory" step, but single-model/single-use, not a shared, cross-instance long-term KB.

- **AutoCompressors: Adapting Language Models to Compress Contexts** (Chevalier, Wettig, Ajith, Chen, EMNLP 2023, arXiv:2305.14788). What: recursively compresses long documents, segment by segment, into "summary vectors" usable as soft prompts by later segments/other tasks. Finding: extends effective context, summary vectors substitute for plain-text ICL demonstrations. Relevance: supports viability of compressing documents into a reusable latent (not raw-text) representation consumed downstream, similar in spirit to storing a document as compressed KV for reuse by another instance.

- **ICAE: In-context Autoencoder for Context Compression** (Ge, Hu, Wang, Wang, Chen, Wei — Microsoft, ICLR 2024, arXiv:2307.06945). What: LoRA-encoder compresses long context into a small number of memory slots consumed by the (frozen) LLM decoder itself. Finding: 4x compression on Llama with ~1% extra params, better latency/GPU memory. Relevance: strong architectural analogue — encoder produces compact memory-slot representation of a document, consumed later by (in Thinker's case, potentially another) instance; still same-model encoder/decoder, single document at a time, no long-term multi-document indexed KB.

- **xRAG: Extreme Context Compression for RAG with One Token** (arXiv:2405.13792, NeurIPS 2024). What: reinterprets dense retriever document embeddings as a special "modality" token fed into a frozen LLM via a trained bridge, replacing the retrieved text entirely. Finding: >10% average gain across 6 knowledge tasks, ~3.5x FLOPs reduction. Relevance: closest existing example of "a document is turned into a single compact latent code consumed by another (frozen) LLM as long-term-KB-like input" — but the code comes from an off-the-shelf dense retriever embedding, not from the LLM's own KV memory / special-token mechanism as in Thinker's proposal.

- **Cartridges: Lightweight and General-Purpose Long-Context Representations via Self-Study** (Stanford/Caltech/Buffalo/DeepMind, arXiv:2506.06266, 2025). What: offline-trains a compact per-corpus KV cache ("cartridge") via self-generated synthetic conversation + context-distillation, loaded at inference instead of the full context. Finding: matches in-context learning with 38.6x less memory, 26.4x more throughput. Relevance: **closest prior art to "self-generated KV memory usable as a long-term KB"** — the KV cache is literally trained (not just cached) to represent a document/corpus and is reused across many later queries/instances; still trained per-corpus offline via self-study rather than produced online end-to-end by a single "encode via special token" forward pass, and not framed around hierarchical/domain indexing or multi-document distractor retrieval.

- **Memory³: Language Modeling with Explicit Memory** (arXiv:2407.01178, Jul 2024). What: introduces "explicit memory" as a third memory type (after implicit/parametric and working/KV-context) — sparsified, retrievable memory chunks formed via a two-stage pretraining scheme, cheaper than parameters or full RAG. Finding: a from-scratch 2.4B model with explicit memory beats larger LLMs and RAG baselines with higher decoding speed. Relevance: strongly supports the general thesis (memory as a distinct, cheaper knowledge store), and the two-stage pretraining that forms memory end-to-end is close in spirit to "trained end-to-end" doc-to-memory encoding Thinker proposes; still no domain-hierarchical indexing analysis or multi-instance distractor-KB retrieval setup.

## MoE expert specialization and self-organization (bears on "does routing organize by domain")

- **Mixtral of Experts** (arXiv:2401.04088, 2024) + independent analyses. Finding: Mixtral shows *little* domain specialization — expert activation is close to uniform across most domains/layers (specialization mainly by syntax/position, not topic).
- **OLMoE: Open Mixture-of-Experts Language Models** (Muennighoff et al., arXiv:2409.02060, 2024). Finding: unlike Mixtral, OLMoE shows strong domain and vocabulary specialization (e.g., near-100% specialization for some experts on arXiv-domain text), early router saturation, weak expert co-activation.
- **DeepSeekMoE: Towards Ultimate Expert Specialization** (arXiv:2401.06066, ACL 2024). What: fine-grained expert segmentation + shared-expert isolation explicitly engineered to *increase* specialization. Finding: reaches near-upper-bound MoE performance at ~40% of dense compute, evidencing specialization is a design lever, not automatic.
- **The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise** (Wang, Hayou, Nalisnick, arXiv:2604.09780, 2026). What: shows analytically (routers are linear maps) that expert-usage similarity is fully explained by hidden-state geometric similarity, not a learned "domain" concept; load-balancing loss actively suppresses shared directions. Finding: across 5 pretrained models, and even for the *same* math question across models, expert activation patterns diverge ~60%, i.e. "specialization" is representation-geometry-driven and inconsistent, not a robust, reusable domain map. Relevance: **directly threatens naive versions of V4** — even where specialization is observed (OLMoE), it is an emergent artifact of embedding geometry under a load-balancing objective, not necessarily a stable, predictable, reusable domain index; the degree/consistency of specialization is architecture- and data-dependent (contrast Mixtral vs OLMoE vs DeepSeekMoE), so V4's claim needs its own direct measurement, not an appeal to MoE literature as given.

## Prefetching / offloading expert weights at inference (systems side — the "predictable prefetch" argument)

- **MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving** (arXiv:2401.14361, 2024). What: exploits *temporal locality* of expert activation across a sequence (sequence-level activation tracing) to guide prefetch/caching of offloaded experts. Finding: 3.1–16.7x latency improvement over vLLM/DeepSpeed/Ollama by exploiting this locality. Relevance: **directly relevant precedent that expert-routing locality is real and exploitable for prefetch** even in *standard* (non-Thinker) MoE models — this actually weakens the claim that MoE prefetch is fundamentally "unpredictable," since systems work already extracts usable locality; Thinker's argument needs to be that its recurrent/iteration-wise, front-loaded fetch pattern is *more* predictable/coarser-grained than this per-token temporal locality, not that MoE has none.
- **Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable MoE Inference** (Hwang et al., arXiv:2308.12066, ISCA 2024). What: modifies the routing algorithm itself to predict next-block expert selection one step ahead ("pre-gating"), enabling overlap of expert fetch with compute. Relevance: shows the field already engineers *algorithmic* predictability into MoE routing (rather than relying on emergent domain locality) — a useful contrast: Thinker's claim would be that domain-organized indexing gives predictability "for free" from the training objective, without needing an explicit pre-gate mechanism.
- **Fiddler: CPU-GPU Orchestration for Fast Inference of MoE Models** (Kamahori et al., ICLR 2025, arXiv:2402.07033). What: systems-level CPU/GPU work split for MoE inference minimizing data movement. Relevance: further evidence that MoE inference-time cost is dominated by expert weight movement, i.e. that fetch pattern predictability has direct measurable cost implications — supports that the metric Thinker should report (bytes moved, resident fraction) is exactly what this literature already optimizes for, giving a ready comparison baseline.

## Model merging / hierarchical composition producing domain structure by construction

- **Branch-Train-MiX (BTX)** (Sukhbaatar et al., arXiv:2403.07816, 2024). What: trains separate domain experts independently (branch), then merges their FFNs into MoE layers + finetunes routing (mix). Relevance: BTX creates domain-specialized experts *by construction* (each expert is literally trained on one domain), not by emergent self-organization — useful contrast case showing what deliberately domain-organized routing looks like, vs. what emergent routing (Mixtral/OLMoE) actually produces; if V4's hierarchical index instead relies on emergent organization from a single joint training run, BTX is the "positive control" for what explicit domain structure would achieve.
- **Hierarchical Routing Mixture of Experts (HRME)** (Zhao et al., arXiv:1903.07756, 2019) and general hierarchical-MoE literature. What: tree-structured gating that recursively soft-partitions input space, each non-leaf a data-dependent router. Relevance: pre-dates modern LLM MoE; establishes that tree/hierarchical routing *can* be trained to reflect real data structure (multimodality) in principle, but this is classical mixture-of-experts (regression/classification), not evidence about LLM-scale semantic domain self-organization — a structural precedent for the "hierarchical index" architecture, not for the self-organization claim itself.

## Conclusions

**(a) Is V4 ("self-organization by domain") supported by literature?** Mixed and unsettled, not a clean "yes." Evidence for: OLMoE shows real, measurable domain/vocabulary specialization emerging from ordinary training (arXiv:2409.02060); DeepSeekMoE shows specialization can be strongly amplified by architecture choices (fine-grained experts + shared-expert isolation, arXiv:2401.06066); MoE-Infinity shows the resulting locality is real and systems-exploitable (arXiv:2401.14361). Evidence against/complicating: Mixtral shows near-zero domain specialization under a very similar training recipe (arXiv:2401.04088), and "The Myth of Expert Specialization" (arXiv:2604.09780) shows analytically that apparent specialization is an artifact of hidden-state geometry under load-balancing pressure, not a robust or portable "domain map" — the same question activates ~60%-divergent expert sets across models. So self-organization by domain is *possible but not guaranteed*, is highly sensitive to architecture/training details, and where it appears is not obviously stable/reusable in the way V4's "predictable prefetch" argument needs. This must be measured directly for Thinker's own indexed memory, not assumed from MoE analogies.

**(b) Is the prefetch/inference argument novel, and what experiment would show it?** Not fully novel as a systems observation — MoE-Infinity, Pre-gated MoE, and Fiddler already establish and exploit locality/predictability in expert access for prefetch/offloading in ordinary MoE models. What would be novel is the specific claim that Thinker's *recurrent-iteration, front-loaded, domain-level* fetch pattern (fetch once early, reuse across iterations) is qualitatively coarser-grained and more predictable than MoE's *per-token* routing — this is a genuinely different claim from "MoE has some temporal locality." Measurable experiment: instrument the indexed-memory core across recurrent iterations processing single-domain vs mixed-domain input batches, and report (1) memory resident fraction — how much of the accessed index fits in a small fixed "hot set" derived from iteration-1 fetches; (2) fetch locality across iterations — overlap (Jaccard/IoU) of fetched index entries between iteration t and iteration 1, per domain, per document; (3) bytes moved — actual data transferred if index entries beyond the hot set require fetch from slower storage, compared against an MoE baseline instrumented the same way (e.g. Mixtral/OLMoE per-token expert IDs) under the same task distribution. A convincing result is high iteration-1→iteration-N overlap and low bytes-moved-after-iteration-1 for Thinker's index, contrasted with sustained per-token churn in the MoE baseline.

**(c) Closest prior art to self-generated KV memory trained end-to-end?** **Cartridges (arXiv:2506.06266)** is closest structurally — a KV cache is *trained* (not merely cached) via self-study/context-distillation to represent a corpus, then reused across many later instances/queries, with large memory/throughput wins. **Memorizing Transformers (arXiv:2203.08913)** is closest mechanically for "special-token/KV-write into a queryable store," though same-instance and short-horizon. **ICAE (arXiv:2307.06945)** and **xRAG (arXiv:2405.13792)** are closest for "one model encodes a document into a compact latent object consumed by another (possibly frozen) model," which matches Thinker's cross-instance framing, but neither is trained fully end-to-end jointly with a distractor-laden long-term-KB retrieval task the way Thinker's proposal is. No paper found that combines all three properties simultaneously (self-generated KV via special token + cross-instance long-term KB with distractors + full end-to-end training) — this combination appears to be the gap Thinker's design targets.

## BibTeX

```bibtex
@inproceedings{lample2019large,
  title={Large Memory Layers with Product Keys},
  author={Lample, Guillaume and Sablayrolles, Alexandre and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019},
  eprint={1907.05242},
  archivePrefix={arXiv}
}

@article{berges2024memory,
  title={Memory Layers at Scale},
  author={Berges, Vincent-Pierre and O{\u{g}}uz, Barlas and Haziza, Daniel and Yih, Wen-tau and Zettlemoyer, Luke and Ghosh, Gargi},
  journal={arXiv preprint arXiv:2412.09764},
  year={2024}
}

@inproceedings{khandelwal2020generalization,
  title={Generalization through Memorization: Nearest Neighbor Language Models},
  author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
  eprint={1911.00172},
  archivePrefix={arXiv}
}

@inproceedings{wu2022memorizing,
  title={Memorizing Transformers},
  author={Wu, Yuhuai and Rabe, Markus N. and Hutchins, DeLesley and Szegedy, Christian},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  eprint={2203.08913},
  archivePrefix={arXiv}
}

@inproceedings{borgeaud2022improving,
  title={Improving Language Models by Retrieving from Trillions of Tokens},
  author={Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and Cai, Trevor and Rutherford, Eliza and Millican, Katie and van den Driessche, George and Lespiau, Jean-Baptiste and Damoc, Bogdan and Clark, Aidan and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022},
  eprint={2112.04426},
  archivePrefix={arXiv}
}

@article{wang2025m,
  title={M+: Extending MemoryLLM with Scalable Long-Term Memory},
  author={Wang, Yu and others},
  journal={International Conference on Machine Learning (ICML)},
  year={2025},
  eprint={2502.00592},
  archivePrefix={arXiv}
}

@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  eprint={2501.00663},
  archivePrefix={arXiv}
}

@article{kang2025lm2,
  title={LM2: Large Memory Models},
  author={Kang, Jikun and others},
  journal={arXiv preprint arXiv:2502.06049},
  year={2025}
}

@inproceedings{mu2023gisting,
  title={Learning to Compress Prompts with Gist Tokens},
  author={Mu, Jesse and Li, Xiang Lisa and Goodman, Noah},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023},
  eprint={2304.08467},
  archivePrefix={arXiv}
}

@inproceedings{chevalier2023autocompressors,
  title={Adapting Language Models to Compress Contexts},
  author={Chevalier, Alexis and Wettig, Alexander and Ajith, Anirudh and Chen, Danqi},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023},
  eprint={2305.14788},
  archivePrefix={arXiv}
}

@inproceedings{ge2024ICAE,
  title={In-context Autoencoder for Context Compression in a Large Language Model},
  author={Ge, Tao and Hu, Jing and Wang, Lei and Wang, Xun and Chen, Si-Qing and Wei, Furu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  eprint={2307.06945},
  archivePrefix={arXiv}
}

@inproceedings{cheng2024xrag,
  title={xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token},
  author={Cheng, Xin and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.13792},
  archivePrefix={arXiv}
}

@article{eyuboglu2025cartridges,
  title={Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study},
  author={Eyuboglu, Sabri and others},
  journal={arXiv preprint arXiv:2506.06266},
  year={2025}
}

@article{yang2024memory3,
  title={Memory$^3$: Language Modeling with Explicit Memory},
  author={Yang, Hongkang and others},
  journal={arXiv preprint arXiv:2407.01178},
  year={2024}
}

@article{jiang2024mixtral,
  title={Mixtral of Experts},
  author={Jiang, Albert Q. and others},
  journal={arXiv preprint arXiv:2401.04088},
  year={2024}
}

@article{muennighoff2024olmoe,
  title={OLMoE: Open Mixture-of-Experts Language Models},
  author={Muennighoff, Niklas and others},
  journal={arXiv preprint arXiv:2409.02060},
  year={2024}
}

@inproceedings{dai2024deepseekmoe,
  title={DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models},
  author={Dai, Damai and others},
  booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2024},
  eprint={2401.06066},
  archivePrefix={arXiv}
}

@article{wang2026myth,
  title={The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise},
  author={Wang, Xi and Hayou, Soufiane and Nalisnick, Eric},
  journal={arXiv preprint arXiv:2604.09780},
  year={2026}
}

@article{xue2024moeinfinity,
  title={MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache},
  author={Xue, Leyang and others},
  journal={arXiv preprint arXiv:2401.14361},
  year={2024}
}

@inproceedings{hwang2024pregated,
  title={Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference},
  author={Hwang, Ranggi and Wei, Jianyu and others},
  booktitle={International Symposium on Computer Architecture (ISCA)},
  year={2024},
  eprint={2308.12066},
  archivePrefix={arXiv}
}

@inproceedings{kamahori2025fiddler,
  title={Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models},
  author={Kamahori, Keisuke and Tang, Tian and Gu, Yile and Zhu, Kan and Kasikci, Baris},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  eprint={2402.07033},
  archivePrefix={arXiv}
}

@article{sukhbaatar2024btx,
  title={Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM},
  author={Sukhbaatar, Sainbayar and others},
  journal={arXiv preprint arXiv:2403.07816},
  year={2024}
}

@article{zhao2019hierarchical,
  title={Hierarchical Routing Mixture of Experts},
  author={Zhao, Wenbo and Gao, Yan and Ji, Tingting and Wan, Xiaobo and Ye, Fu and Lin, Guo},
  journal={arXiv preprint arXiv:1903.07756},
  year={2019}
}
```
