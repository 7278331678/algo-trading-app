{{- define "trading.labels" -}}
app.kubernetes.io/part-of: trading-app
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "trading.secretEnv" -}}
- name: {{ .name }}
  valueFrom:
    secretKeyRef:
      name: {{ .secret }}
      key: {{ .key }}
{{- end -}}

{{/* env shared by the API and every batch job */}}
{{- define "trading.backendEnv" -}}
envFrom:
  - secretRef: { name: {{ .Values.secretName }} }
env:
  {{- include "trading.secretEnv" (dict "name" "POSTGRES_PASSWORD" "secret" .Values.secretName "key" "POSTGRES_PASSWORD") | nindent 2 }}
  {{- include "trading.secretEnv" (dict "name" "S3_ACCESS_KEY" "secret" .Values.secretName "key" "MINIO_ROOT_USER") | nindent 2 }}
  {{- include "trading.secretEnv" (dict "name" "S3_SECRET_KEY" "secret" .Values.secretName "key" "MINIO_ROOT_PASSWORD") | nindent 2 }}
  # the app builds the database URL itself and URL-encodes the password
  - { name: POSTGRES_HOST, value: "postgres" }
  - { name: POSTGRES_PORT, value: "5432" }
  - { name: POSTGRES_USER, value: {{ .Values.postgres.user | quote }} }
  - { name: POSTGRES_DB, value: {{ .Values.postgres.database | quote }} }
  - { name: REDIS_URL, value: "redis://redis:6379/0" }
  - { name: S3_ENDPOINT_URL, value: "http://minio:9000" }
  - { name: S3_BUCKET, value: {{ .Values.minio.bucket | quote }} }
  - { name: SMTP_HOST, value: "mailhog" }
  - { name: SMTP_PORT, value: "1025" }
{{- end -}}
