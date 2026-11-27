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
