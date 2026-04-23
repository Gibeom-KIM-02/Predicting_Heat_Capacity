이 파이썬 코드는 머신러닝 학습용 열역학 데이터셋을 만들기 위한 데이터 수집 스크립트입니다. 

처음에는 파이썬의 thermo 패키지에서 열용량(Cp), 녹는점(mp), 끓는점(bp)을 바로 가져오려고 했으나, 
온도나 압력, 물질의 상태 같은 측정 기준이 제각각이거나 추정치가 섞여 있어서 틀린 값이 너무 많았습니다. 
그래서 방향을 바꿔서 thermo 패키지에서는 분자 이름과 CAS 번호만 가져옵니다. 
그리고 이 정보를 바탕으로 PubChem이나 NIST 같은 여러 출처의 물리화학적 실험값들이 모여있는 Chemeo 사이트를 검색해 데이터를 수집합니다. 
Chemeo에서는 상압(1 atm) 기준의 mp와 bp를 찾고, 열용량(Cp)은 고체, 액체, 기체 상태별로 나누어 수집합니다. 

열용량의 경우 온도가 명시되지 않은 데이터는 기본 상온 값으로 간주하여 수집하고, 온도가 명시되어 있는데 상온(298.15K +- 5K) 범위를 벗어나는 데이터는 모델의 오차를 막기 위해 제외했습니다.

원래는 step1에서 딱 2000개를 모으는 걸 목표로 해서, 상온 기준 상태의 Cp 값이 존재하고 mp와 bp가 모두 있는 데이터만 남기려고 했었습니다. 
하지만 지금은 특정 목표 개수를 정해두지 않고, thermo 패키지 안의 Yaws 데이터베이스에 있는 7549개 화합물의 CAS 번호 전체를 일단 다 가져와서 진행하는 걸로 방식을 바꿨습니다.

수천 개의 데이터를 빠르게 모으기 위해 비동기 방식으로 크롤링을 수행하며, 수집된 SMILES 구조식을 RDKit으로 읽어 분자량(MW)과 회전 가능한 결합 수(Rotatable bonds) 같은 특성들도 함께 계산해 줍니다. 

코드가 작동하면 3개의 엑셀 파일이 나옵니다. 
step1 파일은 Chemeo에서 긁어온 단위가 포함된 기초 데이터입니다. (SMILES, mp, bp, Cp)
step2 파일은 여기에 RDKit으로 계산한 분자량과 회전가능 결합 수를 추가한 파일입니다. 
마지막으로 step2_strict 파일은 mp나 bp 값이 비어있는 경우를 제외하고, 끓는점이 녹는점보다 낮은 오류 데이터도 걸러내며, 
상온 25도에서 물질의 상태(고, 액, 기)와 실제 수집된 열용량의 상태가 완벽하게 일치하는 데이터만 남겨둔 파일입니다.

========

This Python script is a data collection tool for building a thermodynamic dataset for machine learning.

Initially, I tried extracting heat capacity (Cp), melting point (mp), and boiling point (bp) directly from the Python thermo package. 
However, there were too many incorrect values because the measurement standards—such as temperature, pressure, and physical state—varied, or because mathematical estimates were mixed in. 

So, I changed the approach to only extract the molecule name and CAS number from the thermo package. 
Based on this information, the script searches and collects data from Chemeo, a site that aggregates physicochemical experimental values from various sources like PubChem and NIST. 
From Chemeo, it retrieves mp and bp measured at standard atmospheric pressure (1 atm), and collects Cp values separated by solid, liquid, and gas phases.

For heat capacity, if the temperature is not explicitly specified, the data is assumed to be at standard room temperature and collected. However, if the temperature is explicitly stated but falls outside the room temperature range (298.15K +- 5K), the data is excluded to prevent modeling errors.

Originally, the goal was to collect exactly 2,000 items in step1, keeping only the data that had a room-temperature Cp value along with both mp and bp. 
However, the approach has now changed. 
Instead of setting a specific target count, **the script pulls all 7,549 CAS numbers available in the thermo package's Yaws database and processes them all.**

To gather thousands of data points quickly, it performs asynchronous web scraping. 
It also reads the collected SMILES strings using RDKit to calculate structural features like molecular weight (MW) and the number of rotatable bonds.

When the code runs, it outputs three Excel files. 
The step1 file contains the foundational data and units scraped from Chemeo (SMILES, mp, bp, Cp).
The step2 file adds the molecular weight and rotatable bond count calculated by RDKit to the step1 data. 
Finally, the step2_strict file excludes cases where mp or bp values are missing, filters out erroneous data where the boiling point is lower than the melting point, and only retains data where the calculated physical state of the substance at 25 degrees Celsius perfectly matches the phase of the collected heat capacity.
