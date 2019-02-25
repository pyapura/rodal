Rodal
==

## Ambiente

Rodal es un **ambiente** desarrollado para cumplir la interface común especificada en Gym, la librería que contiene un banco de problemas para poner a prueba algoritmos de **aprendizaje reforzado**. [Gym][gym] es un producto de [OpenAI][openai], una empresa sin fines de lucro que mediante la investigación se dedica a promover la inteligencia artificial general de manera segura.

[gym]: https://gym.openai.com/
[openai]: https://openai.com/

---

## Instalación

Una vez completada la instalación, y antes de crear una instancia del ambiente con la función `make()`, es necesario importar la libreria gym y el ambiente propiamente dicho:

```python
import gym
import rodal
amb = gym.make('rodal-v0')
```
