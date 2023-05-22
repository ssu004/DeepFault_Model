# Deep Fault Benchmark 코딩 스타일

## 들여쓰기

들여쓰기는 `space`, `4`로 한다.

## 네이밍 규칙

* 변수명: `snake_case`
* 함수명: `snake_case`
* 클래스명: `PascalCase`
* 메소드명: `snake_case`
* private 메소드명: `_snake_case`
* 멤버명: `snake_case`
* private 멤버명: `_snake_case`
* 모듈명: `alllowercasenounderscore`

## 타입 어노테이션

코드 가독성을 위해 작성하는 함수에 타입 어노테이션을 붙인다.

```python
# Good
def sample_data(
    data: np.array,
    sample_length: int,
    shift: int,
    cls_id: int,
    num_class: int,
    one_hot: bool=False,
) -> Tuple[np.ndarray, np.ndarray]:

# Bad
def sample_data(
    data,
    sample_length,
    shift,
    cls_id,
    num_class,
    one_hot,
):
```

## 코드 린터

PEP8을 기반으로 하는 린터 `black`을 활용하여 포매팅한다.

```bash
# Terminal

$ black source.py
```

## 주석 작성

협업자간 가독성을 위해 함수 및 객체에는 `numpy`/`scipy` 스타일 docstring을 작성하도록 함

[numpydoc Style guide](https://numpydoc.readthedocs.io/en/latest/format.html)를 참조할 것