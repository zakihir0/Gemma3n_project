import fs from 'fs-extra';
import path from 'path';

export const codeAnalysis = {
  async analyze(args: {
    path: string;
    analysis_type?: string;
    language?: string;
  }): Promise<any> {
    try {
      const filePath = args.path;
      const analysisType = args.analysis_type || 'structure';
      
      if (!await fs.pathExists(filePath)) {
        throw new Error(`File or directory not found: ${filePath}`);
      }

      const stats = await fs.stat(filePath);
      
      if (stats.isFile()) {
        return await this.analyzeFile(filePath, analysisType, args.language);
      } else {
        return await this.analyzeDirectory(filePath, analysisType);
      }
    } catch (error) {
      throw new Error(`Code analysis failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  },

  async analyzeFile(filePath: string, analysisType: string, language?: string): Promise<any> {
    const content = await fs.readFile(filePath, 'utf8');
    const ext = path.extname(filePath);
    const detectedLanguage = language || this.detectLanguage(ext);
    
    const analysis: any = {
      file: filePath,
      language: detectedLanguage,
      size: content.length,
      lines: content.split('\n').length,
    };

    switch (analysisType) {
      case 'structure':
        analysis.structure = this.analyzeStructure(content, detectedLanguage);
        break;
      case 'dependencies':
        analysis.dependencies = this.analyzeDependencies(content, detectedLanguage);
        break;
      case 'complexity':
        analysis.complexity = this.analyzeComplexity(content, detectedLanguage);
        break;
      case 'imports':
        analysis.imports = this.analyzeImports(content, detectedLanguage);
        break;
      case 'exports':
        analysis.exports = this.analyzeExports(content, detectedLanguage);
        break;
    }

    return {
      content: [
        {
          type: 'text',
          text: `Code analysis for ${filePath}:\n\n${JSON.stringify(analysis, null, 2)}`,
        },
      ],
    };
  },

  async analyzeDirectory(dirPath: string, analysisType: string): Promise<any> {
    // Basic directory analysis
    const analysis = {
      directory: dirPath,
      type: 'directory_analysis',
      summary: 'Directory analysis not yet implemented',
    };

    return {
      content: [
        {
          type: 'text',
          text: `Directory analysis for ${dirPath}:\n\n${JSON.stringify(analysis, null, 2)}`,
        },
      ],
    };
  },

  detectLanguage(extension: string): string {
    const langMap: { [key: string]: string } = {
      '.js': 'javascript',
      '.ts': 'typescript',
      '.jsx': 'javascript',
      '.tsx': 'typescript',
      '.py': 'python',
      '.java': 'java',
      '.cpp': 'cpp',
      '.c': 'c',
      '.cs': 'csharp',
      '.php': 'php',
      '.rb': 'ruby',
      '.go': 'go',
      '.rs': 'rust',
      '.swift': 'swift',
      '.kt': 'kotlin',
    };
    return langMap[extension] || 'unknown';
  },

  analyzeStructure(content: string, language: string): any {
    const lines = content.split('\n');
    const structure: any = {
      functions: [],
      classes: [],
      variables: [],
      comments: 0,
      empty_lines: 0,
    };

    lines.forEach((line, index) => {
      const trimmed = line.trim();
      
      if (!trimmed) {
        structure.empty_lines++;
        return;
      }

      if (this.isComment(trimmed, language)) {
        structure.comments++;
        return;
      }

      // Basic pattern matching for functions and classes
      if (language === 'javascript' || language === 'typescript') {
        if (trimmed.match(/^(function|const|let|var)\s+\w+/)) {
          structure.functions.push({
            name: trimmed.match(/\w+/)?.[0],
            line: index + 1,
          });
        }
        if (trimmed.match(/^class\s+\w+/)) {
          structure.classes.push({
            name: trimmed.match(/class\s+(\w+)/)?.[1],
            line: index + 1,
          });
        }
      }
    });

    return structure;
  },

  analyzeDependencies(content: string, language: string): any {
    const dependencies: any = {
      imports: [],
      requires: [],
      external: [],
      local: [],
    };

    const lines = content.split('\n');
    
    lines.forEach(line => {
      const trimmed = line.trim();
      
      if (language === 'javascript' || language === 'typescript') {
        // import statements
        const importMatch = trimmed.match(/^import\s+.*from\s+['"]([^'"]+)['"]/);
        if (importMatch) {
          const module = importMatch[1];
          dependencies.imports.push(module);
          if (module.startsWith('.')) {
            dependencies.local.push(module);
          } else {
            dependencies.external.push(module);
          }
        }
        
        // require statements
        const requireMatch = trimmed.match(/require\(['"]([^'"]+)['"]\)/);
        if (requireMatch) {
          const module = requireMatch[1];
          dependencies.requires.push(module);
          if (module.startsWith('.')) {
            dependencies.local.push(module);
          } else {
            dependencies.external.push(module);
          }
        }
      }
    });

    return dependencies;
  },

  analyzeComplexity(content: string, language: string): any {
    const lines = content.split('\n');
    let complexity = 1; // Base complexity
    
    lines.forEach(line => {
      const trimmed = line.trim();
      
      // Count decision points
      if (trimmed.includes('if ') || trimmed.includes('else if ')) complexity++;
      if (trimmed.includes('for ') || trimmed.includes('while ')) complexity++;
      if (trimmed.includes('switch ')) complexity++;
      if (trimmed.includes('case ')) complexity++;
      if (trimmed.includes('catch ')) complexity++;
      if (trimmed.includes('&&') || trimmed.includes('||')) complexity++;
    });

    return {
      cyclomatic_complexity: complexity,
      classification: complexity <= 10 ? 'simple' : complexity <= 20 ? 'moderate' : 'complex',
    };
  },

  analyzeImports(content: string, language: string): any {
    return this.analyzeDependencies(content, language).imports;
  },

  analyzeExports(content: string, language: string): any {
    const exports: string[] = [];
    const lines = content.split('\n');
    
    lines.forEach(line => {
      const trimmed = line.trim();
      
      if (language === 'javascript' || language === 'typescript') {
        if (trimmed.startsWith('export ')) {
          exports.push(trimmed);
        }
      }
    });

    return exports;
  },

  isComment(line: string, language: string): boolean {
    if (language === 'javascript' || language === 'typescript') {
      return line.startsWith('//') || line.startsWith('/*') || line.startsWith('*');
    }
    if (language === 'python') {
      return line.startsWith('#');
    }
    if (language === 'java' || language === 'cpp' || language === 'c') {
      return line.startsWith('//') || line.startsWith('/*') || line.startsWith('*');
    }
    return false;
  },
};