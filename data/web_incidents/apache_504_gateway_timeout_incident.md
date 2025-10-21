# Apache 504 Gateway Timeout — Upstream Application Unresponsive

---

## 🧩 Overview

A 504 Gateway Timeout error occurs when Apache, acting as a reverse proxy, cannot receive a timely response from an upstream backend server within the configured timeout period. This error is critical in 3-tier architectures as it indicates a breakdown in communication between the web tier and application tier.

### What is a 504 Gateway Timeout?

A 504 Gateway Timeout is an HTTP status code that indicates the web server (Apache) did not receive a response from an upstream server (backend application) within the allotted time. This differs from a 502 Bad Gateway, which indicates the upstream server returned an invalid response, or a 503 Service Unavailable, which indicates the upstream server is temporarily overloaded.

### Apache mod_proxy Behavior

When Apache's mod_proxy forwards a request to a backend server, it waits for a response based on several configuration parameters:
- **ProxyTimeout**: Maximum time to wait for a response from the backend
- **Timeout**: Maximum time to wait for any I/O operation
- **KeepAliveTimeout**: Time to keep connections alive to backends

### Symptoms Across Infrastructure

**ELB/ALB Symptoms:**
- Increased latency metrics
- 504 error rate spikes
- Backend health check failures
- Connection timeout alerts

**Web Tier Symptoms:**
- Apache error logs showing timeout messages
- Increased worker process utilization
- Connection pool exhaustion
- Memory pressure from pending requests

**Application Tier Symptoms:**
- High CPU utilization
- Database connection pool exhaustion
- Long-running queries or transactions
- Resource contention issues

---

## 📊 Log Samples

### Web Tier (Apache) Logs

```
2024-01-15T10:00:01.123Z [ERROR] [trace_id:req-001-abc123] [request_id:req-001] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] GET /api/orders - 504 Gateway Timeout - 30000ms - Backend service unavailable
2024-01-15T10:00:01.456Z [ERROR] [trace_id:req-002-def456] [request_id:req-002] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] POST /api/checkout - 504 Gateway Timeout - 30000ms - Backend service timeout
2024-01-15T10:00:01.789Z [ERROR] [trace_id:req-003-ghi789] [request_id:req-003] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] GET /api/reports - 504 Gateway Timeout - 30000ms - Backend service unresponsive
2024-01-15T10:00:02.123Z [ERROR] [trace_id:req-004-jkl012] [request_id:req-004] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] GET /api/analytics - 504 Gateway Timeout - 30000ms - Backend service overloaded
2024-01-15T10:00:02.456Z [ERROR] [trace_id:req-005-mno345] [request_id:req-005] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] POST /api/upload - 504 Gateway Timeout - 30000ms - Backend processing timeout
2024-01-15T10:00:02.789Z [ERROR] [trace_id:req-006-pqr678] [request_id:req-006] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] GET /api/import - 504 Gateway Timeout - 30000ms - Backend processing timeout
2024-01-15T10:00:03.123Z [ERROR] [trace_id:req-007-stu901] [request_id:req-007] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] GET /api/backup - 504 Gateway Timeout - 30000ms - Backend service overloaded
2024-01-15T10:00:03.456Z [ERROR] [trace_id:req-008-vwx234] [request_id:req-008] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] GET /api/metrics - 504 Gateway Timeout - 30000ms - Backend timeout
2024-01-15T10:00:03.789Z [ERROR] [trace_id:req-009-yza567] [request_id:req-009] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] GET /api/stream - 504 Gateway Timeout - 30000ms - Backend stream timeout
2024-01-15T10:00:04.123Z [ERROR] [trace_id:req-010-bcd890] [request_id:req-010] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] POST /api/process - 504 Gateway Timeout - 30000ms - Backend service unavailable
2024-01-15T10:00:04.456Z [ERROR] [trace_id:req-011-efg123] [request_id:req-011] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] GET /api/export - 504 Gateway Timeout - 30000ms - Backend export timeout
2024-01-15T10:00:04.789Z [ERROR] [trace_id:req-012-hij456] [request_id:req-012] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] POST /api/sync - 504 Gateway Timeout - 30000ms - Backend sync timeout
2024-01-15T10:00:05.123Z [ERROR] [trace_id:req-013-klm789] [request_id:req-013] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] GET /api/validate - 504 Gateway Timeout - 30000ms - Backend validation timeout
2024-01-15T10:00:05.456Z [ERROR] [trace_id:req-014-nop012] [request_id:req-014] [ELB:frontend-sg] [AZ:us-east-1b] [EC2:i-0123456789abcdef1] PUT /api/update - 504 Gateway Timeout - 30000ms - Backend update timeout
2024-01-15T10:00:05.789Z [ERROR] [trace_id:req-015-qrs345] [request_id:req-015] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcdef0] DELETE /api/cleanup - 504 Gateway Timeout - 30000ms - Backend cleanup timeout
```

### ELB/ALB Logs

```
2024-01-15T10:00:01.123Z [INFO] [ELB:frontend-sg] [AZ:us-east-1a] Backend health check failed for target i-0987654321fedcba0
2024-01-15T10:00:01.456Z [WARN] [ELB:frontend-sg] [AZ:us-east-1b] Connection timeout to backend i-0987654321fedcba1
2024-01-15T10:00:01.789Z [ERROR] [ELB:frontend-sg] [AZ:us-east-1a] Target group health check failure - all targets unhealthy
2024-01-15T10:00:02.123Z [INFO] [ELB:frontend-sg] [AZ:us-east-1b] Auto scaling triggered due to unhealthy targets
2024-01-15T10:00:02.456Z [WARN] [ELB:frontend-sg] [AZ:us-east-1a] High latency detected - 95th percentile > 5000ms
2024-01-15T10:00:02.789Z [ERROR] [ELB:frontend-sg] [AZ:us-east-1b] Target group deregistering unhealthy instances
2024-01-15T10:00:03.123Z [INFO] [ELB:frontend-sg] [AZ:us-east-1a] New instances being registered to target group
2024-01-15T10:00:03.456Z [WARN] [ELB:frontend-sg] [AZ:us-east-1b] Health check grace period expired for new instances
```

### Application Tier Logs

```
2024-01-15T10:00:01.156Z [ERROR] [trace_id:req-001-7bnk7x] [request_id:req-001] [ELB:backend-sg] [AZ:us-east-1a] [EC2:i-0987654321fedcba0] [Service:user-service] Database connection timeout - 43246ms
2024-01-15T10:00:01.489Z [ERROR] [trace_id:req-002-8sl5zr] [request_id:req-002] [ELB:backend-sg] [AZ:us-east-1b] [EC2:i-0987654321fedcba1] [Service:user-service] ServiceUnavailableException: External service unavailable - 2051ms
2024-01-15T10:00:01.822Z [ERROR] [trace_id:req-003-wowqvp] [request_id:req-003] [ELB:backend-sg] [AZ:us-east-1a] [EC2:i-0987654321fedcba0] [Service:user-service] OutOfMemoryError: Java heap space exhausted
2024-01-15T10:00:02.155Z [ERROR] [trace_id:req-004-3x8m9p] [request_id:req-004] [ELB:backend-sg] [AZ:us-east-1b] [EC2:i-0987654321fedcba1] [Service:user-service] DatabaseTransactionTimeoutException: Transaction timeout after 30000ms
2024-01-15T10:00:02.488Z [ERROR] [trace_id:req-005-k2n7q4] [request_id:req-005] [ELB:backend-sg] [AZ:us-east-1a] [EC2:i-0987654321fedcba0] [Service:user-service] ThreadPoolExhaustedException: No available threads in pool
2024-01-15T10:00:02.821Z [ERROR] [trace_id:req-006-9vx8w3] [request_id:req-006] [ELB:backend-sg] [AZ:us-east-1b] [EC2:i-0987654321fedcba1] [Service:user-service] DatabaseConnectionPoolExhaustedException: No available connections
2024-01-15T10:00:03.154Z [ERROR] [trace_id:req-007-5m6n7o] [request_id:req-007] [ELB:backend-sg] [AZ:us-east-1a] [EC2:i-0987654321fedcba0] [Service:user-service] CircuitBreakerOpenException: Circuit breaker is open
2024-01-15T10:00:03.487Z [ERROR] [trace_id:req-008-8p9q0r] [request_id:req-008] [ELB:backend-sg] [AZ:us-east-1b] [EC2:i-0987654321fedcba1] [Service:user-service] ExternalAPITimeoutException: API call timeout after 25000ms
```

---

## 🔍 Root Cause Analysis

### Primary Causes

1. **Backend Service Overload**
   - High CPU utilization preventing timely response processing
   - Memory exhaustion causing garbage collection pauses
   - Thread pool exhaustion blocking request processing
   - Disk I/O bottlenecks affecting application performance

2. **Database Performance Issues**
   - Long-running queries blocking database connections
   - Database connection pool exhaustion
   - Database locks and deadlocks preventing query execution
   - Database server resource constraints

3. **Network Connectivity Problems**
   - High latency between web tier and application tier
   - Network packet loss or congestion
   - DNS resolution delays
   - Firewall or security group misconfigurations

4. **Resource Contention**
   - Disk I/O bottlenecks affecting application performance
   - Network bandwidth limitations
   - Shared resource conflicts between services
   - Container resource limits in Kubernetes environments

### Configuration Issues

1. **Apache Configuration Problems**
   - Inadequate ProxyTimeout settings
   - Insufficient worker process limits
   - Improper KeepAlive configuration
   - Missing proxy error handling

2. **Load Balancer Configuration**
   - Incorrect health check intervals
   - Inadequate timeout settings
   - Poor target group configuration
   - Misconfigured auto-scaling policies

3. **Application Configuration**
   - Inadequate connection pool sizes
   - Missing timeout configurations
   - Poor thread pool configurations
   - Inadequate memory allocation

---

## 🛠️ Troubleshooting Steps

### Immediate Actions

1. **Check Backend Service Health**
   ```bash
   # Check application tier health
   curl -f http://backend-service:8080/health
   
   # Check service metrics
   curl http://backend-service:8080/metrics
   
   # Check service status
   systemctl status backend-service
   ```

2. **Verify Database Connectivity**
   ```bash
   # Test database connection
   mysql -h database-host -u username -p -e "SELECT 1"
   
   # Check connection pool status
   curl http://backend-service:8080/actuator/health/db
   
   # Check database performance
   mysql -h database-host -u username -p -e "SHOW PROCESSLIST"
   ```

3. **Review System Resources**
   ```bash
   # Check CPU and memory usage
   top -p $(pgrep java)
   
   # Check disk I/O
   iostat -x 1
   
   # Check network connectivity
   netstat -an | grep :8080
   ```

### Configuration Validation

1. **Apache Proxy Settings**
   ```apache
   # Verify ProxyTimeout configuration
   ProxyTimeout 30
   Timeout 60
   KeepAliveTimeout 15
   
   # Check proxy configuration
   ProxyPass /api/ http://backend-service:8080/
   ProxyPassReverse /api/ http://backend-service:8080/
   
   # Add error handling
   ProxyErrorOverride On
   ProxyPreserveHost On
   ```

2. **Load Balancer Health Checks**
   ```yaml
   # Verify health check configuration
   HealthCheckPath: /health
   HealthCheckIntervalSeconds: 30
   HealthCheckTimeoutSeconds: 5
   HealthyThresholdCount: 2
   UnhealthyThresholdCount: 3
   ```

---

## 🔧 Resolution Actions

### Short-term Fixes

1. **Increase Timeout Values**
   - Temporarily increase ProxyTimeout to 60 seconds
   - Adjust ELB health check timeouts
   - Implement circuit breaker patterns
   - Add retry logic with exponential backoff

2. **Scale Resources**
   - Add more backend service instances
   - Increase database connection pool size
   - Scale database resources if needed
   - Implement horizontal pod autoscaling

3. **Implement Fallback Mechanisms**
   - Return cached responses for non-critical endpoints
   - Implement graceful degradation
   - Add retry logic with exponential backoff
   - Implement timeout handling in application code

### Long-term Solutions

1. **Performance Optimization**
   - Optimize database queries and indexing
   - Implement caching strategies (Redis, Memcached)
   - Optimize application code and algorithms
   - Implement connection pooling optimizations

2. **Architecture Improvements**
   - Implement microservices architecture
   - Add message queues for asynchronous processing
   - Implement horizontal scaling strategies
   - Add read replicas for database load distribution

3. **Monitoring and Alerting**
   - Set up proactive monitoring for timeout conditions
   - Implement automated scaling policies
   - Add performance dashboards and alerting
   - Implement distributed tracing

---

## 📈 Prevention Strategies

### Monitoring and Alerting

1. **Key Metrics to Monitor**
   - Response time percentiles (P95, P99)
   - Error rates by endpoint
   - Backend service health status
   - Database connection pool utilization
   - System resource utilization (CPU, memory, disk, network)

2. **Alert Thresholds**
   - Response time > 5 seconds
   - Error rate > 5%
   - Backend service unhealthy
   - Database connection pool > 80% utilization
   - CPU utilization > 80%

### Capacity Planning

1. **Load Testing**
   - Regular performance testing
   - Stress testing with realistic loads
   - Capacity planning based on growth projections
   - Chaos engineering for resilience testing

2. **Auto-scaling Configuration**
   - CPU-based scaling policies
   - Custom metrics for application-specific scaling
   - Predictive scaling based on historical patterns
   - Multi-metric scaling policies

### Best Practices

1. **Configuration Management**
   - Use configuration management tools
   - Implement infrastructure as code
   - Regular configuration audits
   - Environment-specific configurations

2. **Deployment Practices**
   - Blue-green deployments
   - Canary deployments for critical changes
   - Automated rollback procedures
   - Health check validation during deployments

---

## 🔄 Recovery Procedures

### Automated Recovery

1. **Circuit Breaker Implementation**
   ```java
   @CircuitBreaker(name = "backend-service", fallbackMethod = "fallbackMethod")
   public String callBackendService() {
       return restTemplate.getForObject("/api/data", String.class);
   }
   
   public String fallbackMethod(Exception ex) {
       return "Service temporarily unavailable";
   }
   ```

2. **Retry Logic**
   ```java
   @Retryable(value = {ConnectTimeoutException.class}, maxAttempts = 3, backoff = @Backoff(delay = 1000))
   public String callExternalService() {
       return externalServiceClient.getData();
   }
   ```

### Manual Recovery Steps

1. **Service Restart**
   ```bash
   # Restart backend services
   systemctl restart backend-service
   
   # Verify service health
   curl http://backend-service:8080/health
   
   # Check logs for errors
   tail -f /var/log/backend-service/application.log
   ```

2. **Database Recovery**
   ```sql
   -- Check for long-running queries
   SELECT * FROM information_schema.processlist 
   WHERE TIME > 30 AND COMMAND != 'Sleep';
   
   -- Kill problematic queries if necessary
   KILL QUERY process_id;
   ```

---

## 📋 Incident Response Checklist

### Immediate Response (0-15 minutes)
- [ ] Acknowledge the incident
- [ ] Check service health endpoints
- [ ] Review error logs
- [ ] Notify stakeholders
- [ ] Begin troubleshooting

### Short-term Response (15-60 minutes)
- [ ] Implement temporary fixes
- [ ] Scale resources if needed
- [ ] Update monitoring alerts
- [ ] Document findings
- [ ] Communicate status updates

### Long-term Response (1-24 hours)
- [ ] Root cause analysis
- [ ] Implement permanent fixes
- [ ] Update runbooks
- [ ] Conduct post-incident review
- [ ] Implement preventive measures

---

## 🎯 Key Performance Indicators (KPIs)

### Response Time Metrics
- **Target P95 Response Time**: < 2 seconds
- **Target P99 Response Time**: < 5 seconds
- **Maximum Acceptable Timeout**: 30 seconds

### Error Rate Metrics
- **Target Error Rate**: < 0.1%
- **Maximum Acceptable Error Rate**: < 1%
- **Timeout Rate Threshold**: < 0.5%

### Availability Metrics
- **Target Uptime**: 99.9%
- **Maximum Acceptable Downtime**: 8.76 hours/year
- **Recovery Time Objective (RTO)**: < 15 minutes
- **Recovery Point Objective (RPO)**: < 5 minutes

This comprehensive incident documentation provides detailed guidance for understanding, troubleshooting, and preventing Apache 504 Gateway Timeout errors in 3-tier web applications, ensuring optimal performance and reliability.



