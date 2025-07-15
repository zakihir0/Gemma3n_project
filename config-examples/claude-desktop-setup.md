# Claude Desktop設定ガイド

## リモートMCPサーバーの設定手順

### 1. Claude Desktopを開く

Claude Desktop アプリケーションを起動します。

### 2. 設定画面に移動

1. メニューから "Settings" または "設定" を選択
2. "Integrations" または "統合" タブをクリック

### 3. MCP統合を追加

1. "Add Integration" または "統合を追加" ボタンをクリック
2. 以下の情報を入力:

#### 基本設定
- **Name (名前)**: `Cursor-Claude MCP Server`
- **Description (説明)**: `File operations and code analysis tools`
- **Server URL**: `http://localhost:3001/mcp`

#### 認証設定（推奨: OAuth）

**Option 1: OAuth 2.0 (推奨)**
- **Authentication Type**: `OAuth 2.0`
- **Client ID**: (設定が必要な場合のみ)
- **Client Secret**: (設定が必要な場合のみ)
- **Authorization URL**: `http://localhost:3001/auth`
- **Token URL**: `http://localhost:3001/token`

**Option 2: No Authentication (開発用)**
- **Authentication Type**: `None`

#### 詳細設定
- **Timeout**: `30000` (30秒)
- **Retry Count**: `3`
- **Enable CORS**: `true`

### 4. 接続テスト

1. "Test Connection" または "接続テスト" ボタンをクリック
2. 成功メッセージが表示されることを確認

### 5. 利用可能なツールの確認

設定完了後、Claude Desktop内で以下のツールが利用可能になります:

#### ファイル操作
```
@read_file path="./example.js"
@write_file path="./new-file.js" content="console.log('Hello');"
@list_files path="./src" recursive=true
@search_files query="function" path="./src"
```

#### プロジェクト管理
```
@get_project_structure path="." max_depth=3
@get_project_info path="."
@find_config_files path="."
```

#### コード解析
```
@analyze_code path="./src/index.js" analysis_type="structure"
@get_function_definitions path="./src/utils.js"
@find_imports path="./src"
@check_syntax path="./src/main.js"
```

## トラブルシューティング

### 接続エラーが発生する場合

1. **MCPサーバーが起動しているか確認**
   ```bash
   curl http://localhost:3001/health
   ```
   
2. **ポート3001が利用可能か確認**
   ```bash
   lsof -i :3001
   ```

3. **ファイアウォール設定を確認**
   - macOS: システム環境設定 > セキュリティとプライバシー > ファイアウォール
   - Windows: Windows Defender ファイアウォール

### 認証エラーが発生する場合

1. **OAuth設定を確認**
   - Client IDとClient Secretが正しく設定されているか
   - リダイレクトURLが正しいか

2. **認証なしで試す**
   - 開発環境では一時的に認証を無効にして接続テスト

### ツールが表示されない場合

1. **Claude Desktopを再起動**
2. **統合を削除して再設定**
3. **MCPサーバーのログを確認**

## セキュリティ注意事項

- **本番環境では必ずOAuth認証を使用**
- **APIキーやシークレットは環境変数で管理**
- **信頼できるネットワークでのみ使用**
- **定期的にアクセスログを確認**

## 追加設定オプション

### HTTPS対応（推奨）

本番環境では HTTPS を使用することを強く推奨します:

```bash
# SSL証明書を用意してサーバーURL を更新
Server URL: https://your-domain.com/mcp
```

### カスタムポート

デフォルトの3001以外のポートを使用する場合:

```bash
# .env ファイルでポートを変更
PORT=8080
```

### CORS設定

特定のドメインからのアクセスのみ許可する場合:

```bash
# .env ファイルでCORS設定を変更
CORS_ORIGINS=https://claude.ai,https://cursor.sh
```