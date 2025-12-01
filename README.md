How to run this project:

1. Create a virtual environment;
2. Pip install dependencies;
3. Run main.py in debug mode


Example of vscode debugger config:

```
{
    "version": "0.2.0",
    "configurations": [
    
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
```
