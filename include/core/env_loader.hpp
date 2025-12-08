// src/core/env_loader.hpp

#ifndef ENV_LOADER_HPP
#define ENV_LOADER_HPP

#include <string>
#include <fstream>
#include  <sstream>
#include <map>
#include <iostream>

class EnvLoader {
private:
    std::map<std::string, std::string> env_vars;
    
    std::string trim(const std::string& str) {
        size_t first = str.find_first_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

public:
    bool load(const std::string& filepath = ".env") {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "WARNING: .env file is not opening" << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));

                // Remove quotes if present
                if (value.size() >= 2 &&
                    ((value.front() == '"' && value.back() == '"') ||
                     (value.front() == '\'' && value.back() == '\''))) {
                        value = value.substr(1, value.size() - 2);
                }

                env_vars[key] = value;
            }
        }
        return true;
    }

    std::string get(const std::string& key, const std::string& default_value = "") const {
        auto it = env_vars.find(key);
        if (it != env_vars.end()) {
            return it->second;
        }

        // Also check system environment variables
        const char* env_val = std::getenv(key.c_str());
        if (env_val != nullptr) {
            return std::string(env_val);
        }

        return default_value;
    }

    bool has(const std::string& key) const {
        return env_vars.find(key) != env_vars.end() ||
                std::getenv(key.c_str()) != nullptr;
    }
};

#endif