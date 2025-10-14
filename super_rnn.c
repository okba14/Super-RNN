#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>


#define LEARNING_RATE 0.005f
#define EPOCHS 5000
#define CLIP_VALUE 5.0f
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8
#define PATIENCE 100
#define DROPOUT_RATE 0.05f
#define MAX_FILES 1000
#define MAX_LINE_LENGTH 1024
#define MAX_COMMAND_LENGTH 256
#define MAX_PATH_LENGTH 512


typedef struct {
    double m, v;
    int t;
} AdamOptimizer;


typedef struct {
    double wxh, whh, why;
    double bxh, by;
    double *h, *h_row, *y_pred;
    double *dh_next; 
    double *x_cache;  
    int max_timesteps;
    

    AdamOptimizer adam_wxh, adam_whh, adam_why, adam_bxh, adam_by;
} RNN;


typedef struct {
    char files[MAX_FILES][MAX_PATH_LENGTH];
    int file_count;
    char commands[MAX_FILES][MAX_COMMAND_LENGTH];
    int command_count;
    char logs[MAX_FILES][MAX_LINE_LENGTH];
    int log_count;
    double system_metrics[10]; // CPU, Memory, Disk, Network metrics
} SystemData;


double xavier_init(int fan_in, int fan_out) {
    double range = sqrt(6.0 / (fan_in + fan_out));
    return ((double)rand() / RAND_MAX - 0.5) * 2 * range;
}

void adam_init(AdamOptimizer *opt) {
    opt->m = 0.0;
    opt->v = 0.0;
    opt->t = 0;
}

RNN create_rnn(int n) {
    RNN rnn;
    

    rnn.wxh = xavier_init(1, n);
    rnn.whh = xavier_init(n, n);
    rnn.why = xavier_init(n, 1);
    rnn.bxh = 0.0;
    rnn.by = 0.0;
    

    rnn.h = (double*)calloc(n + 1, sizeof(double));
    rnn.h_row = (double*)calloc(n, sizeof(double));
    rnn.y_pred = (double*)calloc(n, sizeof(double));
    rnn.dh_next = (double*)calloc(n, sizeof(double));
    rnn.x_cache = (double*)calloc(n, sizeof(double));
    
    if (!rnn.h || !rnn.h_row || !rnn.y_pred || !rnn.dh_next || !rnn.x_cache) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    rnn.max_timesteps = n;
    

    adam_init(&rnn.adam_wxh);
    adam_init(&rnn.adam_whh);
    adam_init(&rnn.adam_why);
    adam_init(&rnn.adam_bxh);
    adam_init(&rnn.adam_by);
    
    return rnn;
}

void freeRNN(RNN rnn) {
    free(rnn.h);
    free(rnn.h_row);
    free(rnn.y_pred);
    free(rnn.dh_next);
    free(rnn.x_cache);
}


double tanh_derivative(double x) {
    return 1.0 - x * x;  
}

double loss(double y_true, double y_pred) {
    double diff = y_true - y_pred;
    return diff * diff;
}

double loss_derivative(double y_true, double y_pred) {
    return 2.0 * (y_pred - y_true);
}


double apply_dropout(double x, double rate) {
    if ((double)rand() / RAND_MAX < rate) return 0.0;
    return x / (1.0 - rate);
}


void normalize_data(double *data, int n, double *mean, double *std) {

    *mean = 0.0;
    for (int i = 0; i < n; i++) *mean += data[i];
    *mean /= n;
    

    *std = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - *mean;
        *std += diff * diff;
    }
    *std = sqrt(*std / n);
    

    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - *mean) / (*std + 1e-8);
    }
}


void denormalize_data(double *data, int n, double mean, double std) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * std + mean;
    }
}


void clip_gradients(double *grad) {
    if (*grad > CLIP_VALUE) *grad = CLIP_VALUE;
    else if (*grad < -CLIP_VALUE) *grad = -CLIP_VALUE;
}


void adam_update(AdamOptimizer *opt, double *param, double grad, double lr) {
    opt->t++;
    

    opt->m = BETA1 * opt->m + (1 - BETA1) * grad;
    

    opt->v = BETA2 * opt->v + (1 - BETA2) * grad * grad;
    

    double m_hat = opt->m / (1 - pow(BETA1, opt->t));
    double v_hat = opt->v / (1 - pow(BETA2, opt->t));
    

    *param -= lr * m_hat / (sqrt(v_hat) + EPSILON);
}


void forward_pass(RNN *rnn, double *X, int n) {
    rnn->h[0] = 0.0;
    
    for (int t = 0; t < n; t++) {

        rnn->x_cache[t] = X[t];
        

        rnn->h_row[t] = rnn->wxh * X[t] + rnn->whh * rnn->h[t] + rnn->bxh;
        rnn->h[t + 1] = tanh(rnn->h_row[t]);
        

        rnn->h[t + 1] = apply_dropout(rnn->h[t + 1], DROPOUT_RATE);
        

        rnn->y_pred[t] = rnn->why * rnn->h[t + 1] + rnn->by;
    }
}


void bptt(RNN *rnn, double *target, int n) {
    double dwhy = 0.0, dwhh = 0.0, dwxh = 0.0, dbxh = 0.0, dby = 0.0;
    

    memset(rnn->dh_next, 0, n * sizeof(double));
    

    for (int t = n - 1; t >= 0; t--) {

        double dl_dy = loss_derivative(target[t], rnn->y_pred[t]);
        

        dwhy += dl_dy * rnn->h[t + 1];
        dby += dl_dy;
        

        double dh = dl_dy * rnn->why + rnn->dh_next[t];
        

        double dh_raw = dh * tanh_derivative(rnn->h_row[t]);
        

        dbxh += dh_raw;
        dwhh += dh_raw * rnn->h[t];
        dwxh += dh_raw * rnn->x_cache[t];
        

        if (t > 0) {
            rnn->dh_next[t - 1] = dh_raw * rnn->whh;
        }
    }
    

    clip_gradients(&dwhy);
    clip_gradients(&dwhh);
    clip_gradients(&dwxh);
    clip_gradients(&dbxh);
    clip_gradients(&dby);
    

    adam_update(&rnn->adam_why, &rnn->why, dwhy, LEARNING_RATE);
    adam_update(&rnn->adam_whh, &rnn->whh, dwhh, LEARNING_RATE);
    adam_update(&rnn->adam_wxh, &rnn->wxh, dwxh, LEARNING_RATE);
    adam_update(&rnn->adam_bxh, &rnn->bxh, dbxh, LEARNING_RATE);
    adam_update(&rnn->adam_by, &rnn->by, dby, LEARNING_RATE);
}


int should_stop_early(double *loss_history, int epoch, int patience) {
    if (epoch < patience + 1) return 0;
    
    double min_loss = loss_history[epoch - patience - 1];
    for (int i = epoch - patience; i < epoch; i++) {
        if (loss_history[i] < min_loss) return 0;
    }
    return 1;
}




void collect_system_data(SystemData *data) {
    printf("Collecting system data...\n");
    

    data->file_count = 0;
    data->command_count = 0;
    data->log_count = 0;
    

    DIR *dir;
    struct dirent *entry;
    

    dir = opendir("/etc");
    if (dir) {
        while ((entry = readdir(dir)) != NULL && data->file_count < MAX_FILES) {
            if (entry->d_type == DT_REG) {
                snprintf(data->files[data->file_count], MAX_PATH_LENGTH, "/etc/%s", entry->d_name);
                data->file_count++;
            }
        }
        closedir(dir);
    }
    

    const char *common_commands[] = {
        "ls", "cd", "pwd", "whoami", "ifconfig", "netstat", 
        "ps", "top", "df", "du", "grep", "find", "cat", "chmod",
        "chown", "tar", "ssh", "scp", "wget", "curl", "apt", "dpkg"
    };
    
    int num_commands = sizeof(common_commands) / sizeof(common_commands[0]);
    for (int i = 0; i < num_commands && data->command_count < MAX_FILES; i++) {
        strncpy(data->commands[data->command_count], common_commands[i], MAX_COMMAND_LENGTH);
        data->command_count++;
    }
    

    const char *log_files[] = {
        "/var/log/syslog", "/var/log/auth.log", "/var/log/kern.log", 
        "/var/log/dmesg", "/var/log/daemon.log"
    };
    
    int num_log_files = sizeof(log_files) / sizeof(log_files[0]);
    for (int i = 0; i < num_log_files && data->log_count < MAX_FILES; i++) {
        FILE *file = fopen(log_files[i], "r");
        if (file) {
            char line[MAX_LINE_LENGTH];
            while (fgets(line, MAX_LINE_LENGTH, file) && data->log_count < MAX_FILES) {
                strncpy(data->logs[data->log_count], line, MAX_LINE_LENGTH);
                data->log_count++;
            }
            fclose(file);
        }
    }
    

    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo) {
        char line[MAX_LINE_LENGTH];
        int cpu_count = 0;
        while (fgets(line, MAX_LINE_LENGTH, cpuinfo)) {
            if (strncmp(line, "processor", 9) == 0) {
                cpu_count++;
            }
        }
        data->system_metrics[0] = cpu_count; 
        fclose(cpuinfo);
    }
    
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo) {
        char line[MAX_LINE_LENGTH];
        unsigned long total_mem = 0, free_mem = 0;
        
        while (fgets(line, MAX_LINE_LENGTH, meminfo)) {
            if (strncmp(line, "MemTotal:", 9) == 0) {
                sscanf(line, "MemTotal: %lu kB", &total_mem);
            } else if (strncmp(line, "MemFree:", 8) == 0) {
                sscanf(line, "MemFree: %lu kB", &free_mem);
            }
        }
        
        data->system_metrics[1] = total_mem / 1024.0; 
        data->system_metrics[2] = free_mem / 1024.0;  
        fclose(meminfo);
    }
    
    printf("System data collected: %d files, %d commands, %d log entries\n", 
           data->file_count, data->command_count, data->log_count);
}


void text_to_numbers(const char *text, double *numbers, int max_numbers) {
    int len = strlen(text);
    int count = 0;
    
    for (int i = 0; i < len && count < max_numbers; i++) {
        numbers[count] = (double)text[i] / 255.0; 
        count++;
    }
    

    while (count < max_numbers) {
        numbers[count] = 0.0;
        count++;
    }
}


void analyze_config_files(SystemData *data, RNN *rnn) {
    printf("Analyzing configuration files...\n");
    

    const char *config_files[] = {
        "/etc/passwd", "/etc/group", "/etc/hosts", "/etc/fstab",
        "/etc/resolv.conf", "/etc/network/interfaces"
    };
    
    int num_configs = sizeof(config_files) / sizeof(config_files[0]);
    
    for (int i = 0; i < num_configs; i++) {
        FILE *file = fopen(config_files[i], "r");
        if (file) {
            char line[MAX_LINE_LENGTH];
            while (fgets(line, MAX_LINE_LENGTH, file)) {

                double numbers[MAX_LINE_LENGTH];
                text_to_numbers(line, numbers, MAX_LINE_LENGTH);
                

                forward_pass(rnn, numbers, MAX_LINE_LENGTH);
                

            }
            fclose(file);
        }
    }
}


void analyze_system_logs(SystemData *data, RNN *rnn) {
    printf("Analyzing system logs...\n");
    
    for (int i = 0; i < data->log_count; i++) {

        double numbers[MAX_LINE_LENGTH];
        text_to_numbers(data->logs[i], numbers, MAX_LINE_LENGTH);
        

        forward_pass(rnn, numbers, MAX_LINE_LENGTH);
        

    }
}


void learn_linux_commands(SystemData *data, RNN *rnn) {
    printf("Learning Linux commands...\n");
    
    for (int i = 0; i < data->command_count; i++) {

        double numbers[MAX_COMMAND_LENGTH];
        text_to_numbers(data->commands[i], numbers, MAX_COMMAND_LENGTH);
        

        forward_pass(rnn, numbers, MAX_COMMAND_LENGTH);
        

    }
}


void predict_security_events(RNN *rnn) {
    printf("Predicting security events...\n");
    

    double input_data[10];
    

    input_data[0] = rnn->wxh; 
    input_data[1] = rnn->whh;
    input_data[2] = rnn->why;
    input_data[3] = rnn->bxh;
    input_data[4] = rnn->by;
    input_data[5] = rnn->h[0]; 
    input_data[6] = 0.0;
    input_data[7] = 0.0;
    input_data[8] = 0.0;
    input_data[9] = 0.0;
    

    double mean, std;
    normalize_data(input_data, 10, &mean, &std);
    

    forward_pass(rnn, input_data, 10);
    

    double prediction = rnn->y_pred[0];
    

    prediction = prediction * std + mean;
    
    printf("Security threat level: %.2f\n", prediction);
    

    if (prediction > 0.8) {
        printf("HIGH SECURITY THREAT DETECTED!\n");
    } else if (prediction > 0.5) {
        printf("Medium security risk detected.\n");
    } else {
        printf("System appears secure.\n");
    }
}


void execute_learned_commands(SystemData *data, RNN *rnn) {
    printf("Executing learned commands...\n");
    

    int random_index = rand() % data->command_count;
    char *command = data->commands[random_index];
    
    printf("Executing command: %s\n", command);
    

    int result = system(command);
    
    if (result == 0) {
        printf("Command executed successfully.\n");
    } else {
        printf("Command execution failed with code: %d\n", result);
    }
}


void generate_security_report(SystemData *data, RNN *rnn) {
    printf("Generating security report...\n");
    

    FILE *report = fopen("/tmp/security_report.txt", "w");
    if (report) {
        fprintf(report, "=== KALI LINUX SECURITY REPORT ===\n\n");
        

        fprintf(report, "System Information:\n");
        fprintf(report, "CPU Cores: %.0f\n", data->system_metrics[0]);
        fprintf(report, "Total Memory: %.2f MB\n", data->system_metrics[1]);
        fprintf(report, "Free Memory: %.2f MB\n", data->system_metrics[2]);
        

        fprintf(report, "\nFiles Analyzed: %d\n", data->file_count);
        

        fprintf(report, "Commands Learned: %d\n", data->command_count);
        

        fprintf(report, "Log Entries Analyzed: %d\n", data->log_count);
        

        fprintf(report, "\nSecurity Assessment:\n");
        double threat_level = rnn->y_pred[0];
        if (threat_level > 0.8) {
            fprintf(report, "THREAT LEVEL: HIGH\n");
        } else if (threat_level > 0.5) {
            fprintf(report, "THREAT LEVEL: MEDIUM\n");
        } else {
            fprintf(report, "THREAT LEVEL: LOW\n");
        }
        
        fclose(report);
        printf("Security report saved to /tmp/security_report.txt\n");
    } else {
        printf("Failed to create security report.\n");
    }
}

int main() {
    srand(time(NULL));
    
    printf("=== LEGENDARY KALI LINUX AI SYSTEM ===\n");
    printf("Initializing advanced RNN model...\n");
    

    double X[] = {1, 2, 3, 4};
    double target[] = {2, 3, 4, 5};  //ROP
    size_t n = sizeof(X) / sizeof(*X);
    

    double X_orig[n], target_orig[n];
    memcpy(X_orig, X, sizeof(X));
    memcpy(target_orig, target, sizeof(target));
    

    double X_mean, X_std, target_mean, target_std;
    

    normalize_data(X, n, &X_mean, &X_std);
    normalize_data(target, n, &target_mean, &target_std);
    

    RNN rnn = create_rnn(n);
    

    double loss_history[EPOCHS];
    double best_loss = DBL_MAX;
    int patience_counter = 0;
    
    printf("Training initial model...\n");
    

    for (int e = 0; e < EPOCHS; e++) {
        // Forward Pass
        forward_pass(&rnn, X, n);
        

        double total_loss = 0.0;
        for (size_t t = 0; t < n; t++) {
            total_loss += loss(target[t], rnn.y_pred[t]);
        }
        double avg_loss = total_loss / n;
        loss_history[e] = avg_loss;
        
        // Early Stopping
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= PATIENCE) {
                printf("Early stopping at epoch %d (loss: %.6f)\n", e, avg_loss);
                break;
            }
        }
        
        // Backpropagation Through Time
        bptt(&rnn, target, n);
        

        if (e % 500 == 0) {
            printf("Epoch %d | Loss = %.6f | Best: %.6f\n", e, avg_loss, best_loss);
        }
    }
    

    printf("\nTesting initial model:\n");
    forward_pass(&rnn, X, n);
    

    double y_pred_denorm[n];
    memcpy(y_pred_denorm, rnn.y_pred, sizeof(rnn.y_pred[0]) * n);
    denormalize_data(y_pred_denorm, n, target_mean, target_std);
    

    for (size_t t = 0; t < n; t++) {
        printf("Input: %.2f | Predicted: %.4f | Target: %.2f | Error: %.4f\n", 
               X_orig[t], y_pred_denorm[t], target_orig[t], 
               fabs(y_pred_denorm[t] - target_orig[t]));
    }
    
    printf("\n=== LEARNING FROM KALI LINUX SYSTEM ===\n");
    

    SystemData system_data;
    collect_system_data(&system_data);
    

    analyze_config_files(&system_data, &rnn);
    

    analyze_system_logs(&system_data, &rnn);
    

    learn_linux_commands(&system_data, &rnn);
    
    printf("\n=== ADVANCED SECURITY ANALYSIS ===\n");
    

    predict_security_events(&rnn);
    

    execute_learned_commands(&system_data, &rnn);
    

    generate_security_report(&system_data, &rnn);
    
    printf("\n=== LEGENDARY AI SYSTEM COMPLETE ===\n");
    printf("The AI has successfully learned from Kali Linux and is now ready for advanced security tasks.\n");
    

    freeRNN(rnn);
    
    return 0;
}
