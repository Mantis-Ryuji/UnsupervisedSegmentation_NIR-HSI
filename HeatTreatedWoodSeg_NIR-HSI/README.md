## MAEの学習

---

## セグメンテーション結果

<details><summary><b> train </b></summary>

### 試験条件Matrix



### A: クスノキ

| A1 | A2 | A3 |
|----|----|----|
| <img src="images/labels/A1_cluster_labels_latent_ckm.png"> | <img src="images/labels/A2_cluster_labels_latent_ckm.png"> | <img src="images/labels/A3_cluster_labels_latent_ckm.png"> |

### B: クリ

| B1 | B2 | B3 |
|----|----|----|
| <img src="images/labels/B1_cluster_labels_latent_ckm.png"> | <img src="images/labels/B2_cluster_labels_latent_ckm.png"> | <img src="images/labels/B3_cluster_labels_latent_ckm.png"> |

### C: ヒノキ

| C1 | C2 | C3 |
|----|----|----|
| <img src="images/labels/C1_cluster_labels_latent_ckm.png"> | <img src="images/labels/C2_cluster_labels_latent_ckm.png"> | <img src="images/labels/C3_cluster_labels_latent_ckm.png"> |

### D: マツ

| D1 | D2 | D3 |
|----|----|----|
| <img src="images/labels/D1_cluster_labels_latent_ckm.png"> | <img src="images/labels/D2_cluster_labels_latent_ckm.png"> | <img src="images/labels/D3_cluster_labels_latent_ckm.png"> |

### E: ヤマザクラ

| E1 | E2 | E3 |
|----|----|----|
| <img src="images/labels/E1_cluster_labels_latent_ckm.png"> | <img src="images/labels/E2_cluster_labels_latent_ckm.png"> | <img src="images/labels/E3_cluster_labels_latent_ckm.png"> |

### V: ライムウッド

| V1 | V2 | V3 |
|----|----|----|
| <img src="images/labels/V1_cluster_labels_latent_ckm.png"> | <img src="images/labels/V2_cluster_labels_latent_ckm.png"> | <img src="images/labels/V3_cluster_labels_latent_ckm.png"> |

### W: ハードメープル

| W1 | W2 | W3 |
|----|----|----|
| <img src="images/labels/W1_cluster_labels_latent_ckm.png"> | <img src="images/labels/W2_cluster_labels_latent_ckm.png"> | <img src="images/labels/W3_cluster_labels_latent_ckm.png"> |

### Y: スプルース

| Y1 | Y2 | Y3 |
|----|----|----|
| <img src="images/labels/Y1_cluster_labels_latent_ckm.png"> | <img src="images/labels/Y2_cluster_labels_latent_ckm.png"> | <img src="images/labels/Y3_cluster_labels_latent_ckm.png"> |

### Z: オーク

| Z1 | Z2 | Z3 |
|----|----|----|
| <img src="images/labels/Z1_cluster_labels_latent_ckm.png"> | <img src="images/labels/Z2_cluster_labels_latent_ckm.png"> | <img src="images/labels/Z3_cluster_labels_latent_ckm.png"> |

</details>

<details><summary><b> val </b></summary>

### 試料条件Matrix



### 180℃

| tile1 | tile2 | tile3 |
|----|----|----|
| <img src="images/labels/180C_tile1_cluster_labels_latent_ckm.png"> | <img src="images/labels/180C_tile2_cluster_labels_latent_ckm.png"> | <img src="images/labels/180C_tile3_cluster_labels_latent_ckm.png"> |

</details>

<details><summary><b> test </b></summary>

### 試験条件Matrix



### 160℃

| tile1 | tile2 | tile3 |
|----|----|----|
| <img src="images/labels/160C_tile1_cluster_labels_latent_ckm.png"> | <img src="images/labels/160C_tile2_cluster_labels_latent_ckm.png"> | <img src="images/labels/160C_tile3_cluster_labels_latent_ckm.png"> |

</details>

---

## MAE効果検証

---

同じ試験体ではないものの train と val, test の試験体は非常に似通っておりリークに近いかもしれない。汎化性能を正しく評価できていない可能性がある。