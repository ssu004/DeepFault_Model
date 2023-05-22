# Continual Learning Machine Fault Diagnosis Benchmark

이 코드는 연구재단 2차년도 주제 학술대회 발표 주제인 `Continual Learning System For Machin Fault Diagnosis (가제)`의 베이스라인 코드이다.

# 의존성 라이브러리

아나콘다를 통해 패키지를 설치한다. 다른 가상환경 관리 방법을 써도 파이썬 버전, requirements.txt에 들어 있는 패키지 버전만 잘 맞춰주면 된다. 설치시 pip의 버전은 23.0.1이다.

```bash
conda create -n {가상환경명} python=3.10.6
conda activate {가상환경명}
pip install --upgrade pip
pip install -r requirements.txt
```

# 시작 포인트

`examples` 디렉토리에 코드를 이해할 수 있는 주피터 노트북 파일을 제공한다. 이 코드를 보고 시작하면 된다.