# spg

 ##  Dataset

Download the following datasets and place them under the `data/` directory:

* [IU X-Ray Dataset](https://iuhealth.org/find-medical-services/x-rays)
* [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

Expected directory structure:

```
MWCL/
├── config/
├── data/
│   ├── iu_xray/
│   └── mimic_cxr/
├── models/
├── modules/
│   ├── dataloader/
│   ├── modal/
    ├── ..../
│   ├── weighted/
│   ├── metrics/
│   ├── tokenizer/
│   └── utils/
├── pycocoevalcap/
├── main_train.py
├── main_test.py
├── test_iu_xray/
├── test_mimic_cxr/
├── train_iu_xray/
└── README.md

```
