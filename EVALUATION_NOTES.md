# DCS Model Evaluation Notes

## 1. เป้าหมายของการประเมิน

เป้าหมายของการประเมินคือดูว่า g2 ที่สร้างจากโมเดล DCS ของเรา
ใกล้เคียงกับข้อมูลที่ได้จาก Physical Phantom มากแค่ไหน

เป้าหมายของโปรเจกต์ในตอนนี้คือ:

- จำลองผลของ Physical Phantom
- เปรียบเทียบ Simulation/Model g2 กับ Physical Phantom g2
- วิเคราะห์ความคลาดเคลื่อนของโมเดลในแต่ละ Flow Condition และ SDS

โปรเจกต์นี้ไม่ได้มีเป้าหมายเพื่อยืนยันว่าโมเดลสามารถจำลอง
การไหลเวียนเลือดของมนุษย์จริงได้


---

## 2. ข้อมูล Physical Phantom

ไฟล์ที่ใช้:

`reference_data/Dataset_phantom-intra40.mat`

ตัวแปรหลัก:

### Dataset_set_intra40

Shape:

`(7, 1540)`

ข้อมูลที่ใช้:

- Row 0 = Commanded Shallow Flow
- Row 1 = Commanded Deep Flow
- Row 2 = Measured Shallow Flow
- Row 3 = Measured Deep Flow

ในการ generate model ใช้ค่า Commanded Shallow Flow
และ Commanded Deep Flow เป็น input


### ave_gtau_intra40

Shape:

`(10, 501, 1540)`

ข้อมูล g2 ที่ใช้ในการประเมิน:

- SDS 1 cm = index 0
- SDS 2 cm = index 1
- SDS 3 cm = index 2

แต่ละ g2 curve มี 501 points
ครอบคลุม delay time 0–500 microseconds


---

## 3. โครงสร้าง Dataset

จาก MAT file ที่ใช้อยู่พบว่า:

- จำนวน Physical records = 1540
- จำนวน unique Commanded Shallow/Deep conditions = 133
- โดยทั่วไปแต่ละ condition มี 10 Physical trials

Condition `(0,0)` มีจำนวน records มากกว่าปกติ:

- `(0,0)` = 210 records

ดังนั้นผลของ `(0,0)` จะถูกวิเคราะห์แยกออกจาก condition อื่นด้วย


---

## 4. นิยาม 1 Experiment

กำหนดให้:

**1 Physical Trial = 1 Experiment**

ใน 1 Experiment ประกอบด้วย:

- Commanded Shallow Flow
- Commanded Deep Flow
- Measured Shallow Flow
- Measured Deep Flow
- Physical g2 ของ SDS 1 cm จำนวน 501 points
- Physical g2 ของ SDS 2 cm จำนวน 501 points
- Physical g2 ของ SDS 3 cm จำนวน 501 points

ดังนั้น 1 Experiment มีข้อมูลสำหรับเปรียบเทียบทั้งหมด:

`3 SDS × 501 points = 1503 points`


---

## 5. วิธีเปรียบเทียบ Physical กับ Model

สำหรับแต่ละ Physical Trial:

1. อ่าน Commanded Shallow Flow และ Commanded Deep Flow
2. ใช้ค่าดังกล่าวเป็น input ของ model
3. Generate Model g2 สำหรับ SDS 1, 2 และ 3 cm
4. เปรียบเทียบ Model g2 กับ Raw Physical g2 ของ trial นั้นโดยตรง
5. คำนวณ RMSE และ R²

Physical trials หลาย trial ที่อยู่ใน condition เดียวกัน
จะไม่ถูกนำมาเฉลี่ยเป็น reference curve ก่อนคำนวณ error

แต่ละ Physical Trial จะถูกเปรียบเทียบกับ Model แยกกันก่อน

การ plot หลาย Physical trials พร้อมกันใช้เพื่อดู variability
และรูปแบบของ curve เท่านั้น


---

## 6. Model ปัจจุบัน

Model รับ:

- Shallow Flow
- Deep Flow

และสร้าง g2 สำหรับ:

- SDS 1 cm
- SDS 2 cm
- SDS 3 cm

ปัจจุบันการแปลง Flow เป็น alphaDb ใช้:

`alphaDb = flow × 1e-6 mm²/s`

ค่านี้ยังเป็นสมมติฐานของโมเดลปัจจุบัน
และยังไม่ได้ถือว่าเป็น physical conversion ที่ผ่านการยืนยัน

ผล Evaluation ในเอกสารนี้จึงใช้เป็น Baseline Evaluation
ของโมเดลเวอร์ชันปัจจุบัน


---

## 7. Metrics

### 7.1 RMSE ต่อ SDS

สำหรับ Physical g2 และ Model g2 จำนวน N points:

`RMSE = sqrt(mean((Model - Physical)^2))`

คำนวณแยกสำหรับ:

- SDS 1 cm
- SDS 2 cm
- SDS 3 cm


### 7.2 R²

R² ที่ใช้ในการประเมินปัจจุบันกำหนดเป็น:

`R² = Pearson correlation coefficient²`

ใช้เพื่อดูความสัมพันธ์ของรูปแบบ curve ระหว่าง Physical และ Model

R² ไม่ได้ใช้แทน RMSE เนื่องจาก curve สามารถมีรูปแบบการเปลี่ยนแปลง
ที่สัมพันธ์กันสูง แต่ยังมีความคลาดเคลื่อนของค่าจริงสูงได้

ถ้า Physical หรือ Model curve เป็น constant curve
ซึ่งมี standard deviation เท่ากับ 0
Pearson correlation จะไม่สามารถคำนวณได้

ในกรณีดังกล่าว R² จะถูกบันทึกเป็น `NaN`


### 7.3 RMSE ต่อ Experiment

RMSE ของ 1 Experiment คำนวณจาก squared error
ของทั้ง 3 SDS รวมกัน

จำนวนทั้งหมด:

`3 × 501 = 1503 points`

ดังนั้น:

`Experiment RMSE = sqrt(mean(all 1503 squared errors))`


### 7.4 Overall RMSE

เมื่อมี Experiment RMSE จำนวน n ค่า:

`Overall RMSE = sqrt((RMSE_1² + RMSE_2² + ... + RMSE_n²) / n)`

ใช้สำหรับสรุป performance ของหลาย experiments รวมกัน


---

## 8. Full Dataset Evaluation

ทำ Evaluation กับ Physical records ทั้งหมด:

`1540 experiments`

### 8.1 รวมทุก Condition

Number of experiments:

`1540`

ผล:

- Mean Experiment RMSE = 0.12399191
- Minimum RMSE = 0.02098229
- Maximum RMSE = 0.36112129
- Overall RMSE = 0.15369227

แยกตาม SDS:

| SDS | Overall RMSE | Mean R² |
|---|---:|---:|
| 1 cm | 0.13856262 | 0.60185285 |
| 2 cm | 0.16053230 | 0.50931312 |
| 3 cm | 0.16091527 | 0.50718033 |


---

## 9. ผลของ Condition (0,0)

Condition `(0,0)` มี:

`210 experiments`

ผล:

- Mean Experiment RMSE = 0.33270102
- Overall RMSE = 0.33281486
- Minimum RMSE = 0.31396958
- Maximum RMSE = 0.36112129

แยกตาม SDS:

| SDS | Overall RMSE | Mean R² |
|---|---:|---:|
| 1 cm | 0.24095280 | 0.00000000 |
| 2 cm | 0.35905474 | 0.00000000 |
| 3 cm | 0.38120680 | N/A |

สำหรับ SDS 3 ของ `(0,0)` Model curve มี zero variance
ทำให้ Pearson correlation และ R² ไม่สามารถคำนวณได้

ดังนั้นจึงเก็บค่าเป็น `NaN`
แทนการกำหนดให้เป็น 0


---

## 10. Evaluation เมื่อตัด (0,0) ออกจากการสรุป

เมื่อตัด Commanded Flow `(0,0)` ออก:

Number of experiments:

`1330`

ผล:

- Mean Experiment RMSE = 0.09103785
- Minimum RMSE = 0.02098229
- Maximum RMSE = 0.23374391
- Overall RMSE = 0.09930594

แยกตาม SDS:

| SDS | Overall RMSE | Mean R² |
|---|---:|---:|
| 1 cm | 0.11429795 | 0.69688224 |
| 2 cm | 0.09738493 | 0.58973098 |
| 3 cm | 0.08388779 | 0.50718033 |

ผล `(0,0)` ไม่ได้ถูกลบออกจาก Dataset

แต่รายงานแยกเพื่อดูผลกระทบของ no-flow condition
ต่อ Overall Error


---

## 11. Zero-Flow Analysis

แบ่ง experiments ตามสถานะของ Shallow และ Deep Flow:

| Flow Group | จำนวน Experiments | Overall RMSE |
|---|---:|---:|
| Shallow=0, Deep=0 | 210 | 0.33281486 |
| Shallow=0, Deep>0 | 240 | 0.16253698 |
| Shallow>0, Deep=0 | 240 | 0.08768042 |
| Shallow>0, Deep>0 | 850 | 0.07616198 |

จากผลพบว่า error สูงไม่ได้เกิดเฉพาะ `(0,0)`

เมื่อ Shallow Flow = 0 แต่ Deep Flow > 0
Overall RMSE ยังสูงกว่ากลุ่มที่ Shallow Flow > 0 อย่างชัดเจน

ในทางกลับกัน เมื่อ Shallow Flow > 0 แต่ Deep Flow = 0
Overall RMSE อยู่ใกล้กับกลุ่มที่ทั้ง Shallow และ Deep Flow > 0 มากกว่า

จึงพบ pattern ว่าโมเดลปัจจุบันมีความคลาดเคลื่อนสูง
โดยเฉพาะใน condition ที่ Commanded Shallow Flow = 0


---

## 12. Condition-Level Analysis

จาก Dataset พบ:

`133 unique Commanded Flow conditions`


### Lowest-Error Conditions

| Shallow | Deep | Trials | Overall RMSE |
|---:|---:|---:|---:|
| 1 | 1 | 10 | 0.026470 |
| 5 | 40 | 10 | 0.029053 |
| 1 | 0 | 10 | 0.033554 |
| 5 | 60 | 10 | 0.034727 |
| 5 | 20 | 10 | 0.043288 |


### Highest-Error Conditions excluding (0,0)

| Shallow | Deep | Trials | Overall RMSE |
|---:|---:|---:|---:|
| 0 | 1 | 10 | 0.226937 |
| 0 | 60 | 10 | 0.197844 |
| 0 | 50 | 10 | 0.197572 |
| 0 | 2 | 10 | 0.191597 |
| 0 | 55 | 10 | 0.189798 |
| 0 | 70 | 10 | 0.181988 |
| 0 | 40 | 10 | 0.176531 |
| 0 | 30 | 10 | 0.172521 |
| 0 | 3 | 10 | 0.170062 |
| 0 | 65 | 10 | 0.165592 |

High-error conditions หลังจากไม่นับ `(0,0)`
กระจุกตัวอยู่ที่ Shallow Flow = 0


---

## 13. Representative Curve Comparison

เลือก 3 conditions เพื่อดูรูปแบบของ Physical และ Model curves


### 13.1 Low-Error Condition

`Shallow = 1, Deep = 1`

Overall RMSE:

`0.02646980`

ผลแยกตาม SDS:

- SDS 1 RMSE = 0.0237, Mean R² = 0.996
- SDS 2 RMSE = 0.0079, Mean R² = 1.000
- SDS 3 RMSE = 0.0384, Mean R² = 0.977

Model สามารถติดตาม Physical curves ได้ใกล้เคียง
โดยเฉพาะ SDS 2


### 13.2 Representative Condition

`Shallow = 5, Deep = 20`

Overall RMSE:

`0.04328779`

ผลแยกตาม SDS:

- SDS 1 RMSE = 0.0383, Mean R² = 0.957
- SDS 2 RMSE = 0.0521, Mean R² = 0.880
- SDS 3 RMSE = 0.0380, Mean R² = 0.839

ใน SDS 2 และ SDS 3
Model curve มีแนวโน้ม decay เร็วกว่า Physical curves

โดยเฉพาะช่วงกลางถึงปลายของ delay time


### 13.3 High-Error Condition

`Shallow = 0, Deep = 1`

Overall RMSE:

`0.22693713`

ผลแยกตาม SDS:

- SDS 1 RMSE = 0.2278, Mean R² = 0.952
- SDS 2 RMSE = 0.2648, Mean R² = 0.933
- SDS 3 RMSE = 0.1803, Mean R² = 0.972

Physical curves มี decay ชัดเจน
แต่ Model curve decay ช้ากว่ามาก

แม้ R² จะยังสูงกว่า 0.9
แต่ RMSE สูงมาก

กรณีนี้แสดงให้เห็นว่า R² สูงเพียงอย่างเดียว
ไม่ได้หมายความว่า Model และ Physical curve มีค่าที่ใกล้เคียงกัน


---

## 14. สิ่งที่พบจาก Baseline Evaluation

จาก Evaluation ปัจจุบันพบว่า:

1. Model สามารถสร้าง g2 curve ที่มีรูปแบบใกล้เคียงกับ Physical Phantom
   ได้ดีในบาง Flow Conditions

2. Performance ของ Model แตกต่างกันตาม Shallow/Deep Flow Condition

3. Condition `(0,0)` มี Error สูงกว่ากลุ่มอื่นอย่างชัดเจน

4. Condition ที่ Shallow Flow = 0 และ Deep Flow > 0
   ยังมี Error สูง แม้จะไม่นับ `(0,0)`

5. Condition ที่ Deep Flow = 0 แต่ Shallow Flow > 0
   ไม่ได้มี Error สูงในระดับเดียวกับ Shallow Flow = 0

6. ในบาง conditions Model decay เร็วกว่า Physical curve

7. ใน condition `Shallow=0, Deep=1`
   Model decay ช้ากว่า Physical curve อย่างมาก

8. Physical trials ภายใน condition เดียวกันมี variability
   แต่ high-error pattern ที่พบไม่ได้เกิดจาก Physical trial เพียง trial เดียว

9. RMSE และ R² ให้ข้อมูลคนละด้าน
   จึงควรพิจารณาร่วมกัน


---

## 15. Output Files

Full experiment-level results:

`output/evaluation/full_dataset_evaluation.csv`

Condition-level summary:

`output/evaluation/condition_summary.csv`

Representative plots:

`output/evaluation/plots/`

ประกอบด้วย:

- `low_error_shallow_1_deep_1.png`
- `representative_shallow_5_deep_20.png`
- `high_error_shallow_0_deep_1.png`


---

## 16. สิ่งที่ยังต้องตัดสินใจ

### Evaluation Threshold

ยังไม่ได้กำหนดว่า RMSE เท่าใดจึงถือว่า Model มีความแม่นยำเพียงพอ

ต้องกำหนดเกณฑ์โดยมีเหตุผลรองรับ
ไม่ควรเลือก threshold เพียงเพราะทำให้ผลของโมเดลปัจจุบันผ่าน


### Model Improvement

ยังไม่ได้ปรับ parameter ของ Model จากผล Evaluation รอบนี้

Baseline Evaluation ชุดนี้จะถูกเก็บไว้
เพื่อใช้เปรียบเทียบหากมีการปรับปรุง Model ในภายหลัง

### Evaluation Threshold

ปัจจุบันยังไม่ได้กำหนด Final Acceptance Threshold
แต่เตรียม Proposed Threshold ไว้ 2 แนวทางเพื่อเสนออาจารย์

#### Option 1: Strict Target — RMSE ≤ 0.03

อิงจาก Physical-to-Physical variability ของข้อมูล Phantom

เมื่อไม่นับ condition `(0,0)`:

- Overall Physical-to-Physical RMSE = 0.01470608
- 90th percentile = 0.02715865
- 95th percentile = 0.02939079
- 99th percentile = 0.03337058

ดังนั้น RMSE = 0.03 ถูกเสนอเป็น Strict Target
เนื่องจากใกล้เคียงกับ 95th percentile ของความแตกต่าง
ระหว่าง Physical trials ที่ Flow Condition เดียวกัน

ผลของ Model ภายใต้เกณฑ์นี้:

- Experiments excluding `(0,0)`:
  22 / 1330 = 1.65%
- Conditions excluding `(0,0)`:
  2 / 132 = 1.52%

เกณฑ์นี้แสดงระดับความแม่นยำที่ใกล้เคียงกับ
natural trial-to-trial variability ของ Physical Phantom


#### Option 2: Practical Baseline Target — RMSE ≤ 0.10

เสนอเป็นเกณฑ์ที่ผ่อนจาก Physical measurement variability
เพื่อใช้พิจารณา Baseline Simulation Model

ผล Baseline Model เมื่อไม่นับ `(0,0)`:

- Overall RMSE = 0.09930594
- Experiments with RMSE ≤ 0.10:
  908 / 1330 = 68.27%
- Conditions with RMSE ≤ 0.10:
  89 / 132 = 67.42%

ค่า 0.10 ไม่ได้มาจากมาตรฐานของ Physical Phantom
หรือ threshold ที่กำหนดไว้ใน paper
แต่เป็น Proposed Practical Tolerance สำหรับนำไปหารือกับอาจารย์
ว่าระดับความคลาดเคลื่อนดังกล่าวเหมาะสมกับ scope
ของ Senior Project หรือไม่


#### Intermediate Value

ทดลองตรวจสอบ RMSE ≤ 0.05 เพิ่มเติม:

- Experiments excluding `(0,0)`:
  122 / 1330 = 9.17%
- Conditions excluding `(0,0)`:
  13 / 132 = 9.85%

ค่า 0.05 ยังไม่มีหลักฐานรองรับที่ชัดเจนเท่ากับ 0.03
และไม่ได้ครอบคลุม performance ส่วนใหญ่ของ Baseline Model
จึงยังไม่เลือกเป็น Proposed Threshold หลัก


#### Current Decision

ยังไม่เลือก Final Threshold

จะเสนออาจารย์ 2 แนวทาง:

1. Strict / Physical-variability-based target: RMSE ≤ 0.03
2. Practical baseline target: RMSE ≤ 0.10

แล้วให้อาจารย์พิจารณาว่าเกณฑ์ใดเหมาะสมกับ
scope และ expected accuracy ของโปรเจกต์


---

## 17. ขั้นตอนถัดไป

1. เสนอผล Baseline Evaluation ให้อาจารย์
2. เสนอ Candidate Threshold:
   - Strict Target = RMSE ≤ 0.03
   - Practical Baseline Target = RMSE ≤ 0.10
3. ให้อาจารย์พิจารณา Final Acceptance Threshold
4. หาก Model ยังไม่ผ่านเกณฑ์ที่เลือก ค่อยพิจารณาปรับ Model
5. หากมีผลจาก Model อีกแบบ ให้นำมาเปรียบเทียบ strengths และ limitations