# Controladores exportados para C

Os exports estão separados por veículo:

```text
C_codes/bebop1/
C_codes/bebop2/
```

Cada pasta de modelo normalmente contém:

```text
nn_parameters.h/.c   pesos, biases e limites de normalização
nn_operations.h/.c   implementação da inferência
run_controller.c     executável que recebe o vetor de entrada
test_controller.c    teste simples com entrada fixa
compare_python_c.py  comparação numérica entre PyTorch e C
```

Modelos recorrentes mantêm o estado interno entre chamadas. Use `nn_reset()` no início de cada voo ou quando a memória do controlador precisar ser reiniciada.

## Comparar PyTorch e C

Exemplo para o CfC do Bebop1:

```bash
.venv/bin/python \
  LNN_behavioural_cloning_quadrotor/C_codes/bebop1/CFC/compare_python_c.py
```

## Teste C simples

```bash
cd LNN_behavioural_cloning_quadrotor/C_codes/bebop1/CFC
gcc -std=c99 -Wall -Wextra \
  test_controller.c nn_operations.c nn_parameters.c \
  -lm -o test_controller
./test_controller
```

## Simular um export

```bash
.venv/bin/python -m \
  LNN_behavioural_cloning_quadrotor.simulators.Simulator_gazebo_square_C \
  --c-model-dir LNN_behavioural_cloning_quadrotor/C_codes/bebop1/CFC \
  --plot-actions \
  --plot-signals
```

Os plots do simulador C são gravados em `organized_plots/sl_runs/generated/c_controller/`.
