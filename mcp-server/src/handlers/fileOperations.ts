import fs from 'fs-extra';
import path from 'path';
import { glob } from 'glob';

export const fileOperations = {
  async read(args: { path: string }): Promise<any> {
    try {
      const content = await fs.readFile(args.path, 'utf8');
      return {
        content: [
          {
            type: 'text',
            text: `File: ${args.path}\n\n${content}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to read file ${args.path}: ${error instanceof Error ? error.message : String(error)}`);
    }
  },

  async write(args: { path: string; content: string; encoding?: string }): Promise<any> {
    try {
      const dir = path.dirname(args.path);
      await fs.ensureDir(dir);
      await fs.writeFile(args.path, args.content, args.encoding as BufferEncoding || 'utf8');
      
      return {
        content: [
          {
            type: 'text',
            text: `Successfully wrote to ${args.path}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to write file ${args.path}: ${error instanceof Error ? error.message : String(error)}`);
    }
  },

  async list(args: { 
    path?: string; 
    recursive?: boolean; 
    include_hidden?: boolean; 
    pattern?: string;
  }): Promise<any> {
    try {
      const searchPath = args.path || '.';
      const options: any = {
        cwd: searchPath,
        dot: args.include_hidden || false,
      };

      let pattern = args.pattern || '**/*';
      if (!args.recursive) {
        pattern = '*';
      }

      const files = await glob(pattern, options);
      const result = await Promise.all(
        files.map(async (file) => {
          const fullPath = path.join(searchPath, file);
          const stats = await fs.stat(fullPath);
          return {
            name: file,
            path: fullPath,
            type: stats.isDirectory() ? 'directory' : 'file',
            size: stats.isFile() ? stats.size : undefined,
            modified: stats.mtime.toISOString(),
          };
        })
      );

      return {
        content: [
          {
            type: 'text',
            text: `Files in ${searchPath}:\n\n${JSON.stringify(result, null, 2)}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to list files: ${error instanceof Error ? error.message : String(error)}`);
    }
  },

  async search(args: {
    query: string;
    path?: string;
    file_pattern?: string;
    case_sensitive?: boolean;
    max_results?: number;
  }): Promise<any> {
    try {
      const searchPath = args.path || '.';
      const pattern = args.file_pattern || '**/*';
      const files = await glob(pattern, { cwd: searchPath, nodir: true });
      
      const results: any[] = [];
      const regex = new RegExp(
        args.query,
        args.case_sensitive ? 'g' : 'gi'
      );

      for (const file of files) {
        if (results.length >= (args.max_results || 100)) break;
        
        try {
          const fullPath = path.join(searchPath, file);
          const content = await fs.readFile(fullPath, 'utf8');
          const lines = content.split('\n');
          
          lines.forEach((line, index) => {
            if (regex.test(line) && results.length < (args.max_results || 100)) {
              results.push({
                file: fullPath,
                line: index + 1,
                content: line.trim(),
                match: line.match(regex)?.[0],
              });
            }
          });
        } catch {
          // Skip files that can't be read as text
        }
      }

      return {
        content: [
          {
            type: 'text',
            text: `Search results for "${args.query}":\n\n${JSON.stringify(results, null, 2)}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Search failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  },
};