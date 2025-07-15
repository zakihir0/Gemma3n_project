import { Tool } from '@modelcontextprotocol/sdk/types.js';

export const codeAnalysisTools: Tool[] = [
  {
    name: 'analyze_code',
    description: 'Analyze code structure, dependencies, and complexity',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path to the file or directory to analyze',
        },
        analysis_type: {
          type: 'string',
          enum: ['structure', 'dependencies', 'complexity', 'imports', 'exports'],
          description: 'Type of analysis to perform',
          default: 'structure',
        },
        language: {
          type: 'string',
          description: 'Programming language (auto-detected if not specified)',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'get_function_definitions',
    description: 'Extract function definitions from a code file',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path to the code file',
        },
        include_body: {
          type: 'boolean',
          description: 'Whether to include function body in the result',
          default: false,
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'find_imports',
    description: 'Find all import statements in a file or directory',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path to search for imports',
        },
        import_type: {
          type: 'string',
          enum: ['local', 'external', 'all'],
          description: 'Type of imports to find',
          default: 'all',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'check_syntax',
    description: 'Check syntax validity of a code file',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path to the code file to check',
        },
        language: {
          type: 'string',
          description: 'Programming language (auto-detected if not specified)',
        },
      },
      required: ['path'],
    },
  },
];