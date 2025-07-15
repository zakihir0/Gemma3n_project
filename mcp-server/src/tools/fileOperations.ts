import { Tool } from '@modelcontextprotocol/sdk/types.js';

export const fileOperationsTools: Tool[] = [
  {
    name: 'read_file',
    description: 'Read the contents of a file',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The path to the file to read',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'write_file',
    description: 'Write content to a file',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The path to the file to write',
        },
        content: {
          type: 'string',
          description: 'The content to write to the file',
        },
        encoding: {
          type: 'string',
          description: 'File encoding (default: utf8)',
          default: 'utf8',
        },
      },
      required: ['path', 'content'],
    },
  },
  {
    name: 'list_files',
    description: 'List files and directories in a given path',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'The directory path to list (default: current directory)',
          default: '.',
        },
        recursive: {
          type: 'boolean',
          description: 'Whether to list files recursively',
          default: false,
        },
        include_hidden: {
          type: 'boolean',
          description: 'Whether to include hidden files and directories',
          default: false,
        },
        pattern: {
          type: 'string',
          description: 'Glob pattern to filter files (e.g., "*.js", "**/*.ts")',
        },
      },
    },
  },
  {
    name: 'search_files',
    description: 'Search for text content within files',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The text to search for',
        },
        path: {
          type: 'string',
          description: 'The directory to search in (default: current directory)',
          default: '.',
        },
        file_pattern: {
          type: 'string',
          description: 'File pattern to limit search (e.g., "*.js", "*.ts")',
        },
        case_sensitive: {
          type: 'boolean',
          description: 'Whether the search should be case sensitive',
          default: false,
        },
        max_results: {
          type: 'number',
          description: 'Maximum number of results to return',
          default: 100,
        },
      },
      required: ['query'],
    },
  },
];