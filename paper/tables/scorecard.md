| Model | Canary type | Min. copies (gray, none) | Survives dedup? | Survives NFKC? | Survives paraphrase? | Works text-only? | False accusations (clean model) |
|---|---|---|---|---|---|---|---|
| Pythia-160M | Random seq. | 1 | no | yes (1) | yes (4) | no | 0/8 |
|  | Fluent trap | 4 | no | yes (4) | yes (4) | yes (64) | 0/8 |
|  | Watermark | 1 | no | yes (4) | yes (1) | yes (16) | 0/8 |
|  | Fictional fact | 16 | yes (16) | yes (16) | yes (16) | no | 0/8 |
| Pythia-410M | Random seq. | 1 | no | yes (1) | yes (1) | yes (4) | 0/8 |
|  | Fluent trap | 1 | no | yes (1) | yes (1) | yes (4) | 0/8 |
|  | Watermark | 1 | yes (1) | yes (4) | yes (1) | yes (4) | 0/8 |
|  | Fictional fact | 1 | yes (1) | yes (1) | yes (1) | no | 0/8 |

'yes (N)': detectable at p<0.01 from N copies onward (every dose >= N; gray box unless the column says text-only) after that pipeline. Stable rule = exploratory; the pre-registered rule is Fig 1.
