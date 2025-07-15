import { Tool } from '@modelcontextprotocol/sdk/types.js';

export const projectTools: Tool[] = [
  {
    name: 'get_project_structure',
    description: 'Get the structure of the current project',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The root path of the project (default: current directory)',
          default: '.',
        },
        max_depth: {
          type: 'number',
          description: 'Maximum depth to traverse (default: 5)',
          default: 5,
        },
        include_files: {
          type: 'boolean',
          description: 'Whether to include files in the structure',
          default: true,
        },
        ignore_patterns: {
          type: 'array',
          items: {
            type: 'string',
          },
          description: 'Patterns to ignore (e.g., node_modules, .git)',
          default: ['node_modules', '.git', 'dist', 'build', '.next'],
        },
      },
    },
  },
  {
    name: 'get_project_info',
    description: 'Get project information from package.json, README, etc.',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The root path of the project (default: current directory)',
          default: '.',
        },
      },
    },
  },
  {
    name: 'find_config_files',
    description: 'Find configuration files in the project',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The root path to search (default: current directory)',
          default: '.',
        },
        config_types: {
          type: 'array',
          items: {
            type: 'string',
          },
          description: 'Types of config files to find',
          default: ['package.json', 'tsconfig.json', '.eslintrc*', 'webpack*', 'vite*'],
        },
      },
    },
  },
];