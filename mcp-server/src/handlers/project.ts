import fs from 'fs-extra';
import path from 'path';
import { glob } from 'glob';

export const projectOperations = {
  async structure(args: {
    path?: string;
    max_depth?: number;
    include_files?: boolean;
    ignore_patterns?: string[];
  }): Promise<any> {
    try {
      const rootPath = args.path || '.';
      const maxDepth = args.max_depth || 5;
      const ignorePatterns = args.ignore_patterns || ['node_modules', '.git', 'dist', 'build', '.next'];

      const buildTree = async (currentPath: string, depth: number = 0): Promise<any> => {
        if (depth > maxDepth) return null;

        const stats = await fs.stat(currentPath);
        const name = path.basename(currentPath);

        if (ignorePatterns.some(pattern => name.includes(pattern))) {
          return null;
        }

        if (stats.isFile()) {
          return args.include_files ? {
            name,
            type: 'file',
            path: currentPath,
            size: stats.size,
          } : null;
        }

        if (stats.isDirectory()) {
          const children = await fs.readdir(currentPath);
          const childNodes = await Promise.all(
            children.map(child => 
              buildTree(path.join(currentPath, child), depth + 1)
            )
          );

          return {
            name,
            type: 'directory',
            path: currentPath,
            children: childNodes.filter(node => node !== null),
          };
        }

        return null;
      };

      const tree = await buildTree(rootPath);

      return {
        content: [
          {
            type: 'text',
            text: `Project structure:\n\n${JSON.stringify(tree, null, 2)}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get project structure: ${error instanceof Error ? error.message : String(error)}`);
    }
  },

  async info(args: { path?: string }): Promise<any> {
    try {
      const rootPath = args.path || '.';
      const info: any = {
        path: rootPath,
        type: 'unknown',
      };

      // Check for package.json (Node.js project)
      const packageJsonPath = path.join(rootPath, 'package.json');
      if (await fs.pathExists(packageJsonPath)) {
        const packageJson = await fs.readJson(packageJsonPath);
        info.type = 'node';
        info.package = {
          name: packageJson.name,
          version: packageJson.version,
          description: packageJson.description,
          dependencies: Object.keys(packageJson.dependencies || {}),
          devDependencies: Object.keys(packageJson.devDependencies || {}),
          scripts: packageJson.scripts,
        };
      }

      // Check for README files
      const readmeFiles = await glob('README*', { cwd: rootPath, nocase: true });
      if (readmeFiles.length > 0) {
        const readmePath = path.join(rootPath, readmeFiles[0]);
        const readmeContent = await fs.readFile(readmePath, 'utf8');
        info.readme = {
          file: readmeFiles[0],
          preview: readmeContent.slice(0, 500) + (readmeContent.length > 500 ? '...' : ''),
        };
      }

      // Check for other project types
      if (await fs.pathExists(path.join(rootPath, 'Cargo.toml'))) {
        info.type = 'rust';
      } else if (await fs.pathExists(path.join(rootPath, 'go.mod'))) {
        info.type = 'go';
      } else if (await fs.pathExists(path.join(rootPath, 'requirements.txt')) || 
                 await fs.pathExists(path.join(rootPath, 'pyproject.toml'))) {
        info.type = 'python';
      }

      return {
        content: [
          {
            type: 'text',
            text: `Project information:\n\n${JSON.stringify(info, null, 2)}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get project info: ${error instanceof Error ? error.message : String(error)}`);
    }
  },

  async findConfigFiles(args: {
    path?: string;
    config_types?: string[];
  }): Promise<any> {
    try {
      const rootPath = args.path || '.';
      const configTypes = args.config_types || [
        'package.json',
        'tsconfig.json',
        '.eslintrc*',
        'webpack*',
        'vite*',
        '.env*',
        'docker*',
        '.gitignore',
      ];

      const configFiles: any[] = [];

      for (const pattern of configTypes) {
        const files = await glob(pattern, { cwd: rootPath, dot: true });
        for (const file of files) {
          const fullPath = path.join(rootPath, file);
          const stats = await fs.stat(fullPath);
          configFiles.push({
            name: file,
            path: fullPath,
            type: path.extname(file) || 'config',
            size: stats.size,
            modified: stats.mtime.toISOString(),
          });
        }
      }

      return {
        content: [
          {
            type: 'text',
            text: `Configuration files found:\n\n${JSON.stringify(configFiles, null, 2)}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to find config files: ${error instanceof Error ? error.message : String(error)}`);
    }
  },
};