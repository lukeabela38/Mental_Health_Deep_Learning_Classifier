# Deep Learning for Detecting Mental Health Disorders using Social Media Generated Content Deployment

To launch deployment container:

```bash
docker-compose up
```

To check system is functioning:

```bash
curl http://localhost:8000/
```


To send query, where ```<TEXT>``` represents your desired query.

```bash
curl http://localhost:8000/read -d "text=<TEXT>"
```

To launch GUI using Gradio:

```bash
curl http://localhost:8000/gui
```
