# Redis in Cloud-Native Environments Workshop

## 🎯 Workshop Files

This directory contains all the materials needed for the Redis workshop:

### 📚 Main Workshop Materials
- **`workshop.md`** - Complete step-by-step workshop guide for participants
- **`workshop-setup.sh`** - Automated setup script for workshop preparation

### 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   ./workshop-setup.sh
   ```

2. **Start the workshop:**
   ```bash
   cd workshop-files
   ./quick-test.sh  # Verify everything works
   ```

3. **Follow the workshop guide:**
   ```bash
   # Open workshop.md in your editor or browser
   open workshop.md
   ```

### 📋 Workshop Structure

#### Part 1: Docker Compose (45 minutes)
- Redis as external state store
- Data structures and patterns
- Configuration separation
- Monitoring and debugging

#### Part 2: Kubernetes (30 minutes)
- Deploying to Kubernetes
- ConfigMaps and Secrets
- Scaling and high availability
- Redis in cloud-native environments

### 🧪 Hands-On Exercises

The workshop includes several hands-on exercises:
- Redis data structure analysis
- Configuration testing
- Persistence verification
- Scaling demonstrations

### 🎯 Learning Objectives

By the end of the workshop, participants will understand:
- How Redis works as an external state store
- Redis data structures for time-series applications
- Configuration management in cloud-native apps
- Deploying Redis-backed applications to Kubernetes
- Monitoring and debugging Redis in containers

### 🧹 Cleanup

After the workshop:
```bash
cd workshop-files
./cleanup.sh
```

### 📞 Support

If you encounter issues during the workshop:
1. Check the instructor notes for common problems
2. Verify all prerequisites are installed
3. Check Docker Desktop is running with Kubernetes enabled
4. Ensure ports 5001 and 30080 are available

### 🎁 Bonus Content

The workshop includes optional advanced topics:
- Redis Streams and Pub/Sub
- Horizontal Pod Autoscaler
- Production considerations
- Security and monitoring

---
